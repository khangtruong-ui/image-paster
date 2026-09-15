"""Base abstraction for object segmentation and candidate quality evaluation."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image


@dataclass
class SegmentationResult:
    """Result of segmenting an object from an image."""
    object_name: str
    original_image: np.ndarray  # HxWx3 (RGB) or HxWx4 (RGBA)
    mask: np.ndarray            # HxW uint8 (0 or 255)
    extracted_rgba: np.ndarray  # HxWx4 (RGB + Alpha)
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    score: float = 1.0
    rejected: bool = False
    rejection_reason: Optional[str] = None

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def area_ratio(self) -> float:
        if self.mask is None or self.mask.size == 0:
            return 0.0
        total = self.mask.shape[0] * self.mask.shape[1]
        return float(np.count_nonzero(self.mask > 0)) / float(total) if total > 0 else 0.0

    @property
    def width(self) -> int:
        return max(1, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(1, self.bbox[3] - self.bbox[1])

    def to_dict(self) -> dict:
        return {
            "object_name": self.object_name,
            "bbox": self.bbox,
            "score": float(self.score),
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
            "area": self.area,
            "area_ratio": self.area_ratio,
            "width": self.width,
            "height": self.height,
        }


class Segmenter(ABC):
    """Abstract base class for object segmenters."""

    def __init__(
        self,
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.95,
        min_confidence: float = 0.4,
    ):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_confidence = min_confidence

    @abstractmethod
    def segment(
        self,
        image: Union[np.ndarray, str, Path],
        prompt: str,
        score_threshold: Optional[float] = None,
    ) -> SegmentationResult:
        """Segment the requested object from image."""
        pass

    def detect_in_background(
        self,
        background_image: Union[np.ndarray, str, Path],
        prompt: str,
    ) -> Optional[SegmentationResult]:
        """Detect and segment a target object to be overridden/replaced inside a background image.

        Args:
            background_image: Background image canvas.
            prompt: Semantic name of object to detect in background (e.g. 'human', 'person', 'car').

        Returns:
            SegmentationResult if detected and valid; None if not detected or rejected.
        """
        try:
            res = self.segment(background_image, prompt=prompt)
            if not res.rejected and res.area > 0 and res.area_ratio < 0.90:
                return res
            return None
        except Exception:
            return None

    def evaluate_mask(self, mask: np.ndarray, score: float = 1.0) -> Tuple[bool, Optional[str]]:
        """Evaluate if segmented mask is appropriate or should be rejected.

        Returns:
            Tuple of (is_valid, rejection_reason).
        """
        if mask is None or mask.size == 0:
            return False, "Mask is empty"

        total_pixels = mask.shape[0] * mask.shape[1]
        fg_pixels = int(np.count_nonzero(mask > 0))
        area_ratio = fg_pixels / total_pixels

        if fg_pixels == 0:
            return False, "No foreground pixels segmented"

        if area_ratio < self.min_area_ratio:
            return False, f"Mask area ratio ({area_ratio:.3f}) is below minimum threshold ({self.min_area_ratio})"

        if area_ratio > self.max_area_ratio:
            return False, f"Mask area ratio ({area_ratio:.3f}) exceeds maximum threshold ({self.max_area_ratio}); covers whole image"

        if score < self.min_confidence:
            return False, f"Confidence score ({score:.3f}) below threshold ({self.min_confidence})"

        return True, None

    @staticmethod
    def load_image_rgb(image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """Load image as uint8 RGB numpy array."""
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGBA")
            rgba = np.array(pil_img)
            return rgba
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            return image
        raise ValueError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def extract_rgba_from_mask(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Combine image and binary mask into RGBA numpy array."""
        h, w = mask.shape[:2]
        if image_rgb.shape[:2] != (h, w):
            image_rgb = cv2.resize(image_rgb, (w, h))

        if image_rgb.shape[2] == 4:
            # If already RGBA, combine existing alpha with new mask
            rgb = image_rgb[:, :, :3]
            alpha = np.minimum(image_rgb[:, :, 3], mask)
        else:
            rgb = image_rgb[:, :, :3]
            alpha = mask

        rgba = np.dstack([rgb, alpha])
        return rgba

    @staticmethod
    def get_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Compute tight bounding box (x_min, y_min, x_max, y_max) from binary mask."""
        coords = cv2.findNonZero(mask)
        if coords is None:
            return (0, 0, mask.shape[1], mask.shape[0])
        x, y, w, h = cv2.boundingRect(coords)
        return (x, y, x + w, y + h)
