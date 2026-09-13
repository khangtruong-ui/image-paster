"""Mask operations, morphological smoothing, and occlusion intersection analysis."""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2


def clean_mask(mask: np.ndarray, open_k: int = 3, close_k: int = 5) -> np.ndarray:
    """Clean mask boundaries with morphological opening and closing."""
    if mask is None or mask.size == 0:
        return mask
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)
    return cleaned


def feather_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Apply soft Gaussian feathering to mask edges."""
    if radius <= 0:
        return mask
    ksize = 2 * radius + 1
    feathered = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=radius / 2.0)
    return feathered


def compute_canvas_occlusions(
    object_masks: Dict[str, np.ndarray],  # name -> full canvas (H, W) uint8 mask
    z_indices: Dict[str, int],
) -> Dict[str, Dict[str, Any]]:
    """Compute exact pixel-level occlusion statistics for all objects on the canvas.

    Returns:
        Dict mapping object_name -> {
            "total_pixels": int,
            "visible_pixels": int,
            "occluded_pixels": int,
            "visibility_ratio": float,
            "occluded_by": Dict[str, int] # occluder_name -> pixel count
        }
    """
    sorted_names = sorted(object_masks.keys(), key=lambda n: z_indices.get(n, 0))
    results: Dict[str, Dict[str, Any]] = {}

    # Iterate from back to front
    for idx, name in enumerate(sorted_names):
        curr_mask = object_masks[name]
        total = int(np.count_nonzero(curr_mask > 0))
        if total == 0:
            results[name] = {
                "total_pixels": 0,
                "visible_pixels": 0,
                "occluded_pixels": 0,
                "visibility_ratio": 1.0,
                "occluded_by": {},
            }
            continue

        # Front objects are all objects with higher index
        front_names = sorted_names[idx + 1 :]
        cum_occlusion = np.zeros_like(curr_mask, dtype=bool)
        occluded_by: Dict[str, int] = {}

        for f_name in front_names:
            f_mask = object_masks[f_name]
            intersection = (curr_mask > 0) & (f_mask > 0)
            inter_count = int(np.count_nonzero(intersection))
            if inter_count > 0:
                occluded_by[f_name] = inter_count
                cum_occlusion |= intersection

        occluded_pixels = int(np.count_nonzero(cum_occlusion))
        visible_pixels = max(0, total - occluded_pixels)
        vis_ratio = visible_pixels / total if total > 0 else 1.0

        results[name] = {
            "total_pixels": total,
            "visible_pixels": visible_pixels,
            "occluded_pixels": occluded_pixels,
            "visibility_ratio": float(vis_ratio),
            "occluded_by": occluded_by,
        }

    return results
