"""End-to-end Semantic Image Generator pipeline with research-oriented trace logging."""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any, List
import cv2
import numpy as np
from PIL import Image

from image_paster.dsl import parse_dsl, SceneIR
from image_paster.llm.planner import BaseScenePlanner, RuleBasedPlanner
from image_paster.retrieval.base import ImageRetriever, ImageCandidate, RetrievalResult
from image_paster.retrieval.duckduckgo import DuckDuckGoRetriever
from image_paster.retrieval.mock import MockRetriever
from image_paster.segmentation.base import Segmenter, SegmentationResult
from image_paster.segmentation.sam3 import SAM3Segmenter
from image_paster.scene.layout import SemanticLayoutSolver, LayoutPlan
from image_paster.rendering.compositor import SceneCompositor, CompositeResult
from image_paster.verification.verifier import SceneVerifier, SemanticVisualVerifier, VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Final result of the semantic image generation pipeline."""
    image_rgb: np.ndarray
    image_bgr: np.ndarray
    dsl_text: str
    scene_ir: SceneIR
    layout_plan: LayoutPlan
    verification: VerificationResult
    execution_trace: Dict[str, Any]
    output_image_path: Optional[str] = None
    output_dsl_path: Optional[str] = None
    output_trace_path: Optional[str] = None

    def save(
        self,
        image_path: str | Path = "output.png",
        dsl_path: str | Path = "generated_scene.dsl",
        trace_path: str | Path = "execution_trace.json",
    ) -> None:
        """Save image, DSL, and execution trace to files."""
        img_p = Path(image_path)
        dsl_p = Path(dsl_path)
        tr_p = Path(trace_path)

        img_p.parent.mkdir(parents=True, exist_ok=True)
        dsl_p.parent.mkdir(parents=True, exist_ok=True)
        tr_p.parent.mkdir(parents=True, exist_ok=True)

        # Save Image
        Image.fromarray(self.image_rgb).save(str(img_p))
        self.output_image_path = str(img_p)

        # Save DSL
        dsl_p.write_text(self.dsl_text, encoding="utf-8")
        self.output_dsl_path = str(dsl_p)

        # Save Trace JSON
        with open(tr_p, "w", encoding="utf-8") as f:
            json.dump(self.execution_trace, f, indent=2)
        self.output_trace_path = str(tr_p)


class SemanticImageGenerator:
    """End-to-end Diffusion-Free Semantic Image Generator.

    Orchestrates: Prompt -> LLM -> C++ DSL -> Retrieval -> SAM3 Segmentation
                  -> Semantic Layout -> OpenCV Compositing -> Poisson Blending
                  -> Verification -> Retry Loop -> Final Image + Trace.
    """

    def __init__(
        self,
        planner: Optional[BaseScenePlanner] = None,
        retriever: Optional[ImageRetriever] = None,
        segmenter: Optional[Segmenter] = None,
        layout_solver: Optional[SemanticLayoutSolver] = None,
        compositor: Optional[SceneCompositor] = None,
        verifier: Optional[SceneVerifier] = None,
        max_retries: int = 2,
    ):
        self.planner = planner or RuleBasedPlanner()
        # Fall back to mock retriever if DDG hits rate limits
        mock_retriever = MockRetriever()
        self.retriever = retriever or DuckDuckGoRetriever(fallback_retriever=mock_retriever)
        self.segmenter = segmenter or SAM3Segmenter()
        self.layout_solver = layout_solver or SemanticLayoutSolver()
        self.compositor = compositor or SceneCompositor(default_blend_mode="poisson")
        self.verifier = verifier or SemanticVisualVerifier()
        self.max_retries = max_retries

    def generate(
        self,
        prompt: str,
        dsl_override: Optional[str] = None,
        blend_mode: Optional[str] = None,
        background_image: Optional[np.ndarray] = None,
    ) -> GenerationResult:
        """Run the full generation pipeline."""
        trace: Dict[str, Any] = {
            "prompt": prompt,
            "pipeline_stages": [],
            "retrieval": {},
            "segmentation": {},
            "layout": {},
            "rendering": {},
            "verification": {},
            "retry_history": [],
        }

        # 1. Scene Planning (LLM / Rule-Based) -> C++ DSL
        if dsl_override:
            dsl_text = dsl_override
            scene_ir = parse_dsl(dsl_text, validate=True)
            planner_type = "override"
        else:
            dsl_text, scene_ir = self.planner.plan(prompt)
            planner_type = self.planner.__class__.__name__

        trace["planner"] = planner_type
        trace["dsl"] = dsl_text
        trace["scene_ir"] = scene_ir.to_dict()
        trace["pipeline_stages"].append("planning")

        # 2. Image Retrieval
        retrieval_results: Dict[str, RetrievalResult] = {}
        for name, obj_ir in scene_ir.objects.items():
            res = self.retriever.retrieve(
                object_name=name,
                source_reqs=obj_ir.source,
                appearance=obj_ir.appearance,
                max_results=3,
            )
            retrieval_results[name] = res
            trace["retrieval"][name] = res.to_dict()

        trace["pipeline_stages"].append("retrieval")

        # 3. Object Segmentation with SAM 3 & candidate rejection
        segmentations: Dict[str, SegmentationResult] = {}
        extracted_sizes: Dict[str, tuple[int, int]] = {}

        for name, res in retrieval_results.items():
            seg_for_object = None
            for cand in res.candidates:
                # Ensure image is locally cached/available
                img_path = cand.local_cached_path
                if not img_path:
                    img_path = self.retriever.download_image(cand)

                if not img_path:
                    continue

                # Run SAM 3 segmentation
                seg_res = self.segmenter.segment(img_path, prompt=name)
                if not seg_res.rejected:
                    seg_for_object = seg_res
                    break
                else:
                    logger.debug(f"Candidate {cand.ranking} for {name} rejected: {seg_res.rejection_reason}")

            # If all candidates rejected or none succeeded, use last or mock fallback
            if seg_for_object is None:
                if res.candidates and res.candidates[0].local_cached_path:
                    seg_for_object = self.segmenter.segment(res.candidates[0].local_cached_path, prompt=name)
                else:
                    # Synthetic fallback
                    dummy = np.zeros((400, 400, 4), dtype=np.uint8)
                    cv2.circle(dummy, (200, 200), 150, (180, 180, 180, 255), -1)
                    seg_for_object = self.segmenter.segment(dummy, prompt=name)

            segmentations[name] = seg_for_object
            extracted_sizes[name] = (seg_for_object.width, seg_for_object.height)
            trace["segmentation"][name] = {
                "score": float(seg_for_object.score),
                "bbox": seg_for_object.bbox,
                "rejected": seg_for_object.rejected,
                "area": seg_for_object.area,
            }

        trace["pipeline_stages"].append("segmentation")

        # 4. Scene Layout Solving + Compositing + Verification (with Bounded Retry Loop)
        current_scene_ir = scene_ir
        final_composite: Optional[CompositeResult] = None
        final_layout_plan: Optional[LayoutPlan] = None
        final_verification: Optional[VerificationResult] = None

        for attempt in range(self.max_retries + 1):
            # Layout Solving
            layout_plan = self.layout_solver.solve(
                scene=current_scene_ir,
                extracted_sizes=extracted_sizes,
            )
            final_layout_plan = layout_plan

            # Compositing
            composite = self.compositor.render(
                scene=current_scene_ir,
                layout_plan=layout_plan,
                segmentations=segmentations,
                background_image=background_image,
                blend_mode=blend_mode,
            )
            final_composite = composite

            # Visual Verification
            verification = self.verifier.verify(
                scene=current_scene_ir,
                layout_plan=layout_plan,
                composite=composite,
            )
            final_verification = verification

            if verification.passed or attempt == self.max_retries:
                if attempt > 0:
                    trace["retry_history"].append({
                        "attempt": attempt,
                        "status": "PASSED" if verification.passed else "MAX_RETRIES_REACHED",
                        "score": verification.score,
                    })
                break

            # Attempt repair based on verification suggestions
            logger.info(f"Verification failed on attempt {attempt + 1}: {verification.issues}. Applying suggestions...")
            trace["retry_history"].append({
                "attempt": attempt,
                "status": "FAILED",
                "issues": verification.issues,
                "suggestions": verification.suggestions,
                "score": verification.score,
            })

            # Apply repair: adjust object positions in scene IR or layout
            for issue in verification.issues:
                if "does not touch ground" in issue:
                    for name in current_scene_ir.objects.keys():
                        if name in issue:
                            current_scene_ir.objects[name].standing_on = "ground"
                elif "is not behind" in issue:
                    # Switch depth if needed
                    for name in current_scene_ir.objects.keys():
                        if f"'{name}'" in issue and "behind" in issue:
                            current_scene_ir.objects[name].depth = "distant"
                elif "does not occlude" in issue:
                    # Bring occluder to overlap occludee
                    for name in current_scene_ir.objects.keys():
                        if f"'{name}' does not occlude" in issue:
                            current_scene_ir.objects[name].region = "center"

        trace["layout"] = final_layout_plan.to_dict()
        trace["rendering"] = {
            "blend_mode": blend_mode or self.compositor.default_blend_mode,
            "canvas_width": final_layout_plan.canvas_width,
            "canvas_height": final_layout_plan.canvas_height,
            "occlusion_stats": final_composite.occlusion_stats,
        }
        trace["verification"] = final_verification.to_dict()
        trace["final_status"] = "SUCCESS" if final_verification.passed else "COMPLETED_WITH_WARNINGS"

        return GenerationResult(
            image_rgb=final_composite.image_rgb,
            image_bgr=final_composite.image_bgr,
            dsl_text=dsl_text,
            scene_ir=scene_ir,
            layout_plan=final_layout_plan,
            verification=final_verification,
            execution_trace=trace,
        )
