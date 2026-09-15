"""SAM 3 Segmenter implementation using Hugging Face transformers."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Union
import numpy as np
import cv2
from PIL import Image

from image_paster.segmentation.base import Segmenter, SegmentationResult

logger = logging.getLogger(__name__)

# Hugging Face transformers SAM3 imports
SAM3_AVAILABLE = False
try:
    import torch
    from transformers import (
        Sam3Model,
        Sam3Processor,
        Sam3ImageProcessor,
        Sam3Config,
    )
    SAM3_AVAILABLE = True
except ImportError:
    pass


class SAM3Segmenter(Segmenter):
    """Object extraction segmenter based on SAM 3 (Segment Anything Model 3).

    Integrates Hugging Face's `transformers.Sam3Model` and `transformers.Sam3Processor`.
    Supports automatic fallback from `facebook/sam3` to the mirror repository `jetjodh/sam3`,
    as well as CV-based heuristic fallback if offline or checkpoints are unavailable.
    """

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        mirror_model_name: str = "jetjodh/sam3",
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.95,
        min_confidence: float = 0.3,
        force_fallback: bool = False,
    ):
        super().__init__(min_area_ratio, max_area_ratio, min_confidence)
        self.model_name = model_name
        self.mirror_model_name = mirror_model_name
        self.hf_token = hf_token
        self.force_fallback = force_fallback

        if device:
            self.device = device
        elif SAM3_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._image_processor: Optional[Any] = None
        self._active_model_name: Optional[str] = None
        self._init_attempted = False

    @property
    def active_model_name(self) -> Optional[str]:
        """Return the repository name of the active loaded SAM 3 model."""
        return self._active_model_name

    def _load_model(self) -> bool:
        """Attempt to load SAM 3 model from Hugging Face transformers.

        First attempts `self.model_name` (e.g. 'facebook/sam3'), then falls back to
        `self.mirror_model_name` (e.g. 'jetjodh/sam3'). If both fail, falls back to CV heuristic mode.
        """
        if self._init_attempted:
            return self._model is not None

        self._init_attempted = True
        if self.force_fallback or not SAM3_AVAILABLE:
            return False

        models_to_try = [self.model_name]
        if self.mirror_model_name and self.mirror_model_name != self.model_name:
            models_to_try.append(self.mirror_model_name)

        for target_repo in models_to_try:
            try:
                logger.info(f"Attempting to load SAM 3 from '{target_repo}' on {self.device}...")
                self._processor = Sam3Processor.from_pretrained(
                    target_repo,
                    token=self.hf_token,
                )
                dtype = torch.float16 if self.device == "cuda" else torch.float32
                self._model = Sam3Model.from_pretrained(
                    target_repo,
                    token=self.hf_token,
                    torch_dtype=dtype,
                ).to(self.device)
                self._model.eval()
                self._active_model_name = target_repo
                logger.info(f"Successfully loaded SAM 3 from '{target_repo}' on {self.device}")
                return True
            except Exception as e:
                logger.warning(
                    f"SAM 3 repository '{target_repo}' could not be loaded ({type(e).__name__}: {e}). "
                    f"Trying fallback..."
                )

        logger.warning(
            f"All SAM 3 model repositories ({models_to_try}) failed to load. "
            f"Operating in CV fallback segmentation mode."
        )
        try:
            self._image_processor = Sam3ImageProcessor()
        except Exception:
            pass
        return False

    def is_model_loaded(self) -> bool:
        """Check if SAM 3 pretrained model is loaded in memory."""
        self._load_model()
        return self._model is not None

    def segment(
        self,
        image: Union[np.ndarray, str, Path],
        prompt: str,
        score_threshold: Optional[float] = None,
    ) -> SegmentationResult:
        """Segment target object identified by prompt from image."""
        raw_img = self.load_image_rgb(image)
        h, w = raw_img.shape[:2]

        loaded = self._load_model()
        score = 0.9
        mask = None

        if loaded and self._model is not None and self._processor is not None:
            # Try full prompt first
            mask, score = self._segment_with_sam3(raw_img, prompt)
            # If multi-word prompt didn't detect or score was low, try last noun/category word
            if mask is None and " " in prompt.strip():
                simplified_prompt = prompt.strip().split()[-1]
                mask, score = self._segment_with_sam3(raw_img, simplified_prompt)

        if mask is None:
            # High quality fallback segmentation using alpha or GrabCut/salience
            mask, score = self._fallback_segment(raw_img, prompt)

        # Ensure mask is 2D uint8
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 127).astype(np.uint8) * 255

        # Evaluate candidate mask against rejection criteria
        thresh = score_threshold or self.min_confidence
        is_valid, reason = self.evaluate_mask(mask, score)

        extracted_rgba = self.extract_rgba_from_mask(raw_img, mask)
        bbox = self.get_mask_bbox(mask)

        return SegmentationResult(
            object_name=prompt,
            original_image=raw_img,
            mask=mask,
            extracted_rgba=extracted_rgba,
            bbox=bbox,
            score=score,
            rejected=not is_valid,
            rejection_reason=reason,
        )

    def detect_in_background(
        self,
        background_image: Union[np.ndarray, str, Path],
        prompt: str,
    ) -> Optional[SegmentationResult]:
        """Detect and segment a target object inside a background image for override/replacement.

        Uses SAM 3 with textual prompt (e.g. 'human', 'person', 'car') on the background image.
        If SAM 3 finds the object and the mask satisfies area criteria (not full image, not empty),
        returns the SegmentationResult with coordinates to paste the replacement object.
        If no matching object is detected in the background, returns None so the pipeline
        can fall back to normal placement.
        """
        raw_img = self.load_image_rgb(background_image)
        h, w = raw_img.shape[:2]

        loaded = self._load_model()
        mask = None
        score = 0.0

        if loaded and self._model is not None and self._processor is not None:
            mask, score = self._segment_with_sam3(raw_img, prompt)
            if mask is None and " " in prompt.strip():
                simplified = prompt.strip().split()[-1]
                mask, score = self._segment_with_sam3(raw_img, simplified)

        if mask is None:
            # Check fallback heuristic detection in background
            mask, score = self._detect_fallback_in_background(raw_img, prompt)

        if mask is None:
            return None

        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 127).astype(np.uint8) * 255

        is_valid, reason = self.evaluate_mask(mask, score)
        if not is_valid:
            logger.debug(f"Background detection for '{prompt}' rejected: {reason}")
            return None

        # Ensure detected object doesn't dominate > 80% of background (must be a subject in the scene)
        area_ratio = np.count_nonzero(mask > 0) / float(h * w)
        if area_ratio > 0.80 or area_ratio < 0.005:
            return None

        extracted_rgba = self.extract_rgba_from_mask(raw_img, mask)
        bbox = self.get_mask_bbox(mask)

        return SegmentationResult(
            object_name=prompt,
            original_image=raw_img,
            mask=mask,
            extracted_rgba=extracted_rgba,
            bbox=bbox,
            score=score,
            rejected=False,
        )

    def _detect_fallback_in_background(self, image_arr: np.ndarray, prompt: str) -> tuple[Optional[np.ndarray], float]:
        """CV fallback detection for target object in background (saliency/contrast/contours)."""
        h, w = image_arr.shape[:2]
        rgb = image_arr[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # 1. Look for high contrast / foreground salient objects in center or midground
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 40, 140)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_area = 0
        total_area = h * w

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Must be between 0.8% and 60% of background area
            if 0.008 * total_area < area < 0.60 * total_area:
                if area > best_area:
                    best_area = area
                    best_contour = cnt

        if best_contour is not None:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [best_contour], -1, 255, -1)
            return mask, 0.75

        return None, 0.0

    def _segment_with_sam3(self, image_rgb: np.ndarray, prompt: str) -> tuple[Optional[np.ndarray], float]:
        """Perform forward pass with Hugging Face Sam3Model."""
        try:
            pil_image = Image.fromarray(image_rgb[:, :, :3])
            inputs = self._processor(
                images=pil_image,
                text=prompt,
                return_tensors="pt",
            ).to(self.device)

            if self.device == "cuda" and hasattr(self._model, "dtype"):
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Post process masks
            target_sizes = [(image_rgb.shape[0], image_rgb.shape[1])]
            if hasattr(self._processor, "post_process_instance_segmentation"):
                results = self._processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.min_confidence,
                    target_sizes=target_sizes,
                )
                if results and "masks" in results[0] and len(results[0]["masks"]) > 0:
                    scores = results[0].get("scores")
                    if scores is not None and len(scores) > 0:
                        scores_np = scores.cpu().numpy()
                        best_idx = int(np.argmax(scores_np))
                        score_val = float(scores_np[best_idx])
                    else:
                        best_idx = 0
                        score_val = 0.9
                    masks = results[0]["masks"].cpu().numpy()
                    mask_uint8 = (masks[best_idx] > 0).astype(np.uint8) * 255
                    return mask_uint8, score_val

            return None, 0.0
        except Exception as e:
            logger.warning(f"SAM 3 forward pass failed for prompt '{prompt}': {e}")
            return None, 0.0

    def _fallback_segment(self, image_arr: np.ndarray, prompt: str) -> tuple[np.ndarray, float]:
        """Robust CV-based segmentation fallback (alpha mask / GrabCut / Otsu)."""
        h, w = image_arr.shape[:2]

        # 1. If image has alpha channel with non-trivial opacity, use it!
        if image_arr.shape[2] == 4:
            alpha = image_arr[:, :, 3]
            fg_count = np.count_nonzero(alpha > 20)
            if 0.01 * (h * w) < fg_count < 0.98 * (h * w):
                bin_mask = (alpha > 50).astype(np.uint8) * 255
                return bin_mask, 0.95

        # 2. Check if background is nearly solid white / black
        rgb = image_arr[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Check corners to see if background is white
        corner_pixels = np.array([
            gray[:10, :10].mean(),
            gray[:10, -10:].mean(),
            gray[-10:, :10].mean(),
            gray[-10:, -10:].mean(),
        ])
        if corner_pixels.mean() > 235:
            # White background thresholding
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            return mask, 0.88

        if corner_pixels.mean() < 25:
            # Dark background thresholding
            _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            return mask, 0.85

        # 3. Use center-prior GrabCut
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        # Margin around border
        margin_x = max(2, int(w * 0.05))
        margin_y = max(2, int(h * 0.05))
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        try:
            cv2.grabCut(rgb, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
            bin_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")
            return bin_mask, 0.80
        except Exception:
            # Fallback to center ellipse
            bin_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(bin_mask, (w // 2, h // 2), (int(w * 0.4), int(h * 0.4)), 0, 0, 360, 255, -1)
            return bin_mask, 0.70
