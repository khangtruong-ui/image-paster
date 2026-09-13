"""Object segmentation package."""

from image_paster.segmentation.base import (
    Segmenter,
    SegmentationResult,
)
from image_paster.segmentation.sam3 import SAM3Segmenter

__all__ = [
    "Segmenter",
    "SegmentationResult",
    "SAM3Segmenter",
]
