"""Unit tests for SAM 3 object segmentation and candidate rejection."""

import numpy as np
import pytest
from image_paster.segmentation.base import Segmenter, SegmentationResult
from image_paster.segmentation.sam3 import SAM3Segmenter


def test_mask_evaluation_criteria():
    segmenter = SAM3Segmenter()

    # Empty mask
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    valid, reason = segmenter.evaluate_mask(empty_mask, score=0.9)
    assert not valid
    assert "No foreground" in reason or "empty" in reason

    # Huge mask (99% of image)
    huge_mask = np.ones((100, 100), dtype=np.uint8) * 255
    valid, reason = segmenter.evaluate_mask(huge_mask, score=0.9)
    assert not valid
    assert "exceeds maximum threshold" in reason

    # Tiny mask (1 pixel)
    tiny_mask = np.zeros((100, 100), dtype=np.uint8)
    tiny_mask[50, 50] = 255
    valid, reason = segmenter.evaluate_mask(tiny_mask, score=0.9)
    assert not valid
    assert "below minimum threshold" in reason

    # Normal mask (20% of image)
    normal_mask = np.zeros((100, 100), dtype=np.uint8)
    normal_mask[30:70, 30:70] = 255
    valid, reason = segmenter.evaluate_mask(normal_mask, score=0.85)
    assert valid
    assert reason is None


def test_sam3_segmenter_execution_on_synthetic_image():
    # 200x200 white background with green circle in center
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    # Center circle
    for y in range(200):
        for x in range(200):
            if (x - 100) ** 2 + (y - 100) ** 2 < 40 ** 2:
                img[y, x] = [34, 139, 34]

    segmenter = SAM3Segmenter(force_fallback=True)
    res = segmenter.segment(img, prompt="tree")

    assert isinstance(res, SegmentationResult)
    assert res.mask.shape == (200, 200)
    assert not res.rejected
    assert res.extracted_rgba.shape == (200, 200, 4)
    assert res.area > 100
    # Check bbox
    x1, y1, x2, y2 = res.bbox
    assert x1 >= 50 and x2 <= 150
    assert y1 >= 50 and y2 <= 150


def test_sam3_transformers_classes_availability():
    """Verify SAM 3 components from Hugging Face transformers."""
    from transformers import Sam3Config, Sam3ImageProcessor, Sam3Model
    config = Sam3Config()
    assert config is not None
    image_proc = Sam3ImageProcessor()
    assert image_proc is not None
