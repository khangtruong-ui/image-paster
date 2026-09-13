"""Geometric transformations for segmented image objects."""

from __future__ import annotations
from typing import Optional
import numpy as np
import cv2


def transform_object(
    rgba: np.ndarray,
    target_width: int,
    target_height: int,
    rotation: float = 0.0,
    flip_h: bool = False,
    flip_v: bool = False,
    perspective: Optional[str] = None,
) -> np.ndarray:
    """Apply resizing, flipping, rotation, and perspective transforms to RGBA image."""
    h, w = rgba.shape[:2]
    if w <= 0 or h <= 0:
        return np.zeros((max(1, target_height), max(1, target_width), 4), dtype=np.uint8)

    # 1. Flip
    img = rgba.copy()
    if flip_h and flip_v:
        img = cv2.flip(img, -1)
    elif flip_h:
        img = cv2.flip(img, 1)
    elif flip_v:
        img = cv2.flip(img, 0)

    # 2. Resize
    target_w = max(4, target_width)
    target_h = max(4, target_height)
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA if target_w < w else cv2.INTER_LINEAR)

    # 3. Rotation
    if abs(rotation) > 0.1:
        center = (target_w // 2, target_h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, rotation, 1.0)
        img = cv2.warpAffine(img, rot_mat, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # 4. Optional perspective tilt
    if perspective:
        p_mode = perspective.lower()
        src_pts = np.float32([[0, 0], [target_w, 0], [0, target_h], [target_w, target_h]])
        if p_mode in ("tilt_left", "left_angle"):
            dst_pts = np.float32([[0, int(target_h * 0.1)], [target_w, 0], [0, int(target_h * 0.9)], [target_w, target_h]])
            m = cv2.getPerspectiveTransform(src_pts, dst_pts)
            img = cv2.warpPerspective(img, m, (target_w, target_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        elif p_mode in ("tilt_right", "right_angle"):
            dst_pts = np.float32([[0, 0], [target_w, int(target_h * 0.1)], [0, target_h], [target_w, int(target_h * 0.9)]])
            m = cv2.getPerspectiveTransform(src_pts, dst_pts)
            img = cv2.warpPerspective(img, m, (target_w, target_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    return img
