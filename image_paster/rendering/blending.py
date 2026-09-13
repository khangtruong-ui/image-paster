"""Poisson seamless cloning, alpha compositing, and illumination matching."""

from __future__ import annotations
import logging
from typing import Tuple, Optional
import numpy as np
import cv2

from image_paster.rendering.masks import feather_mask

logger = logging.getLogger(__name__)


def adjust_lighting_and_color(
    bgr: np.ndarray,
    temperature: str = "natural",
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
) -> np.ndarray:
    """Adjust color temperature, brightness, contrast, and saturation of BGR image."""
    img = bgr.astype(np.float32)

    # 1. Color temperature adjustment
    temp = temperature.lower()
    if "warm" in temp:
        # Boost Red, reduce Blue
        img[:, :, 2] = np.clip(img[:, :, 2] * 1.08, 0, 255)  # Red
        img[:, :, 1] = np.clip(img[:, :, 1] * 1.02, 0, 255)  # Green
        img[:, :, 0] = np.clip(img[:, :, 0] * 0.92, 0, 255)  # Blue
    elif "cool" in temp:
        # Boost Blue, reduce Red
        img[:, :, 0] = np.clip(img[:, :, 0] * 1.10, 0, 255)  # Blue
        img[:, :, 2] = np.clip(img[:, :, 2] * 0.90, 0, 255)  # Red

    # 2. Contrast and Brightness: output = contrast * input + brightness
    if contrast != 1.0 or brightness != 0.0:
        img = np.clip(contrast * img + brightness, 0, 255)

    # 3. Saturation via HSV
    if saturation != 1.0:
        u8 = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(u8, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)


def alpha_composite(
    background_bgr: np.ndarray,
    foreground_rgba: np.ndarray,
    x: int,
    y: int,
    opacity: float = 1.0,
) -> np.ndarray:
    """Standard alpha compositing with sub-region boundary clamping."""
    bg = background_bgr.copy()
    bg_h, bg_w = bg.shape[:2]
    fg_h, fg_w = foreground_rgba.shape[:2]

    # Compute intersection with canvas
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + fg_w)
    y2 = min(bg_h, y + fg_h)

    if x1 >= x2 or y1 >= y2:
        return bg

    fg_x1 = x1 - x
    fg_y1 = y1 - y
    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)

    fg_sub = foreground_rgba[fg_y1:fg_y2, fg_x1:fg_x2]
    bg_sub = bg[y1:y2, x1:x2]

    fg_rgb = fg_sub[:, :, :3]
    alpha = (fg_sub[:, :, 3].astype(np.float32) / 255.0) * float(opacity)
    alpha = np.expand_dims(alpha, axis=-1)

    blended = (fg_rgb.astype(np.float32) * alpha + bg_sub.astype(np.float32) * (1.0 - alpha))
    bg[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return bg


def poisson_blend(
    background_bgr: np.ndarray,
    foreground_rgba: np.ndarray,
    x: int,
    y: int,
    blend_mode: str = "normal",
) -> np.ndarray:
    """Seamless cloning using OpenCV Poisson blending with boundary safety checks."""
    bg = background_bgr.copy()
    bg_h, bg_w = bg.shape[:2]
    fg_h, fg_w = foreground_rgba.shape[:2]

    # Safety margin for cv2.seamlessClone (requires target region completely inside destination)
    margin = 4
    if x < margin or y < margin or (x + fg_w) > (bg_w - margin) or (y + fg_h) > (bg_h - margin):
        # Fall back to feathered alpha composite to prevent OpenCV assertion crash
        logger.debug("Object touches canvas border; falling back to feathered alpha blend.")
        feathered = foreground_rgba.copy()
        feathered[:, :, 3] = feather_mask(feathered[:, :, 3], radius=4)
        return alpha_composite(bg, feathered, x, y)

    src_bgr = foreground_rgba[:, :, :3].copy()
    src_mask = foreground_rgba[:, :, 3].copy()

    # Pre-clean mask: seamlessClone requires mask boundary to be 0 at edges
    src_mask[0, :] = 0
    src_mask[-1, :] = 0
    src_mask[:, 0] = 0
    src_mask[:, -1] = 0

    if np.count_nonzero(src_mask) == 0:
        return bg

    center = (int(x + fg_w // 2), int(y + fg_h // 2))
    flags = cv2.MIXED_CLONE if blend_mode == "mixed" else cv2.NORMAL_CLONE

    try:
        cloned = cv2.seamlessClone(src_bgr, bg, src_mask, center, flags)
        return cloned
    except Exception as e:
        logger.warning(f"Poisson seamlessClone failed ({e}); falling back to alpha blend.")
        return alpha_composite(bg, foreground_rgba, x, y)
