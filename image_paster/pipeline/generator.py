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
from image_paster.llm.planner import BaseScenePlanner, RuleBasedPlanner, create_llm_planner
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
        creative: bool = True,
        max_retries: int = 2,
        debug: bool = False,
    ):
        self.creative = creative
        self.planner = planner or create_llm_planner(creative=creative)
        # Fall back to mock retriever if DDG hits rate limits
        mock_retriever = MockRetriever()
        self.retriever = retriever or DuckDuckGoRetriever(fallback_retriever=mock_retriever)
        self.segmenter = segmenter or SAM3Segmenter()
        self.layout_solver = layout_solver or SemanticLayoutSolver()
        self.compositor = compositor or SceneCompositor(default_blend_mode="natural")
        self.verifier = verifier or SemanticVisualVerifier()
        self.max_retries = max_retries
        self.debug = debug

    def generate(
        self,
        prompt: str,
        dsl_override: Optional[str] = None,
        blend_mode: Optional[str] = None,
        background_image: Optional[np.ndarray] = None,
        debug: Optional[bool] = None,
        debug_dir: Optional[str | Path] = None,
    ) -> GenerationResult:
        """Run the full generation pipeline."""
        is_debug = self.debug if debug is None else debug
        dbg_path = Path(debug_dir or "debug")
        if is_debug:
            dbg_path.mkdir(parents=True, exist_ok=True)
            print(f"\n[DEBUG] ==================== PIPELINE EXECUTION ====================")
            print(f"[DEBUG] User Prompt: '{prompt}'")
            print(f"[DEBUG] Debug artifacts directory: {dbg_path.resolve()}")

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

        if is_debug:
            print(f"\n[DEBUG:Planning] Planner: {planner_type}")
            print(f"[DEBUG:Planning] Compiled C++ Scene DSL:\n{dsl_text}\n")
            (dbg_path / "00_compiled_scene.dsl").write_text(dsl_text, encoding="utf-8")

        trace["planner"] = planner_type
        trace["dsl"] = dsl_text
        trace["scene_ir"] = scene_ir.to_dict()
        trace["pipeline_stages"].append("planning")

        # 2. Image Retrieval
        # 2a. Background Image Retrieval (if background_image not explicitly provided)
        if background_image is None:
            bg_query = scene_ir.environment.query
            if not bg_query and scene_ir.environment.env_type not in ("studio", "none"):
                bg_query = f"{scene_ir.environment.env_type} landscape background"

            if bg_query:
                from image_paster.dsl.ir import SourceReqsIR
                bg_reqs = SourceReqsIR(query=bg_query)
                bg_res = self.retriever.retrieve(
                    object_name="background",
                    source_reqs=bg_reqs,
                    max_results=3,
                )
                trace["retrieval"]["background"] = bg_res.to_dict()
                if is_debug:
                    is_mock_bg = any(c.image_url.startswith("mock://") for c in bg_res.candidates)
                    source_label_bg = "MOCK / SYNTHETIC" if is_mock_bg else "REAL (DuckDuckGo)"
                    print(f"[DEBUG:Retrieval] Background: query='{bg_res.query}', candidates={len(bg_res.candidates)} [{source_label_bg}]")

                for cand in bg_res.candidates:
                    img_path = cand.local_cached_path
                    if not img_path:
                        img_path = self.retriever.download_image(cand)
                    if img_path and Path(img_path).exists():
                        try:
                            bg_pil = Image.open(img_path).convert("RGB")
                            background_image = np.array(bg_pil)
                            if is_debug:
                                dest_bg = dbg_path / "01_retrieval_background.png"
                                try:
                                    import shutil
                                    shutil.copyfile(img_path, dest_bg)
                                except Exception:
                                    pass
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load background image candidate from {img_path}: {e}")

        # 2b. Foreground Objects Retrieval
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
            if is_debug:
                is_mock = any(c.image_url.startswith("mock://") for c in res.candidates)
                source_label = "MOCK / SYNTHETIC" if is_mock else "REAL (DuckDuckGo)"
                print(f"[DEBUG:Retrieval] Object '{name}': query='{res.query}', candidates={len(res.candidates)} [{source_label}]")
                for idx, cand in enumerate(res.candidates):
                    print(f"      Candidate {cand.ranking}: url='{cand.image_url}'")
                    if cand.local_cached_path and Path(cand.local_cached_path).exists():
                        dest_cand = dbg_path / f"01_retrieval_{name}_{cand.ranking}.png"
                        try:
                            import shutil
                            shutil.copyfile(cand.local_cached_path, dest_cand)
                        except Exception:
                            pass

        trace["pipeline_stages"].append("retrieval")

        # 3. Object Segmentation with SAM 3 & candidate rejection
        segmentations: Dict[str, SegmentationResult] = {}
        extracted_sizes: Dict[str, tuple[int, int]] = {}

        if is_debug:
            active_m = getattr(self.segmenter, "active_model_name", None) or self.segmenter.__class__.__name__
            print(f"\n[DEBUG:Segmentation] Segmenting objects with {active_m}:")

        for name, res in retrieval_results.items():
            seg_for_object = None
            chosen_candidate = None
            obj_ir = scene_ir.objects.get(name)
            seg_prompt = obj_ir.source.query if (obj_ir and obj_ir.source.query) else name.replace("_", " ")

            if is_debug:
                print(f"  --> Segmenting object: '{name}' (prompt='{seg_prompt}')")

            for cand in res.candidates:
                # Ensure image is locally cached/available
                img_path = cand.local_cached_path
                if not img_path:
                    img_path = self.retriever.download_image(cand)

                if not img_path:
                    continue

                # Run SAM 3 segmentation
                seg_res = self.segmenter.segment(img_path, prompt=seg_prompt)
                status_str = "ACCEPTED" if not seg_res.rejected else f"REJECTED ({seg_res.rejection_reason})"
                if is_debug:
                    print(f"      Candidate {cand.ranking}: area={seg_res.area}px, score={seg_res.score:.2f}, bbox={seg_res.bbox} -> {status_str}")

                if not seg_res.rejected:
                    seg_for_object = seg_res
                    chosen_candidate = cand
                    break
                else:
                    logger.debug(f"Candidate {cand.ranking} for {name} rejected: {seg_res.rejection_reason}")

            # If all candidates rejected or none succeeded, use last or mock fallback
            if seg_for_object is None:
                if is_debug:
                    print(f"      Warning: All candidates for '{name}' rejected; applying robust fallback segmentation.")
                if res.candidates and res.candidates[0].local_cached_path:
                    seg_for_object = self.segmenter.segment(res.candidates[0].local_cached_path, prompt=seg_prompt)
                    chosen_candidate = res.candidates[0]
                else:
                    # Synthetic fallback
                    dummy = np.zeros((400, 400, 4), dtype=np.uint8)
                    cv2.circle(dummy, (200, 200), 150, (180, 180, 180, 255), -1)
                    seg_for_object = self.segmenter.segment(dummy, prompt=seg_prompt)

            segmentations[name] = seg_for_object
            extracted_sizes[name] = (seg_for_object.width, seg_for_object.height)
            
            trace["segmentation"][name] = {
                "object_name": name,
                "segmentation_prompt": seg_prompt,
                "segmenter": getattr(self.segmenter, "active_model_name", None) or self.segmenter.__class__.__name__,
                "model_loaded": getattr(self.segmenter, "is_model_loaded", lambda: False)(),
                "candidate_source": chosen_candidate.source_url if chosen_candidate else "synthetic",
                "score": float(seg_for_object.score),
                "bbox": seg_for_object.bbox,
                "rejected": seg_for_object.rejected,
                "rejection_reason": seg_for_object.rejection_reason,
                "area": seg_for_object.area,
                "width": seg_for_object.width,
                "height": seg_for_object.height,
            }

            if is_debug:
                # Save mask and transparent cutout
                mask_file = dbg_path / f"02_segmentation_{name}_mask.png"
                cutout_file = dbg_path / f"02_segmentation_{name}_cutout.png"
                cv2.imwrite(str(mask_file), seg_for_object.mask)
                Image.fromarray(seg_for_object.extracted_rgba).save(str(cutout_file))

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

            if is_debug:
                print(f"\n[DEBUG:Layout & Render] Attempt {attempt + 1}:")
                for o_name, o_layout in layout_plan.objects.items():
                    print(f"  - Object '{o_name}': pos=({o_layout.x}, {o_layout.y}), size=({o_layout.width}x{o_layout.height}), z={o_layout.z_index}")
                print(f"[DEBUG:Verification] Attempt {attempt + 1}: {'PASS' if verification.passed else 'FAIL'} (Score: {verification.score:.2f})")
                if verification.issues:
                    print(f"  Issues detected: {verification.issues}")

                # Save wireframe layout and comparison images
                wireframe = self._draw_layout_wireframe(layout_plan, background_image)
                cv2.imwrite(str(dbg_path / "03_layout_wireframe.png"), cv2.cvtColor(wireframe, cv2.COLOR_RGB2BGR))

                alpha_comp = self.compositor.render(current_scene_ir, layout_plan, segmentations, background_image=background_image, blend_mode="alpha")
                poisson_comp = self.compositor.render(current_scene_ir, layout_plan, segmentations, background_image=background_image, blend_mode="poisson")
                Image.fromarray(alpha_comp.image_rgb).save(str(dbg_path / "04_composite_alpha.png"))
                Image.fromarray(poisson_comp.image_rgb).save(str(dbg_path / "04_composite_poisson.png"))
                Image.fromarray(composite.image_rgb).save(str(dbg_path / "05_composite_final.png"))

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

        if is_debug:
            with open(dbg_path / "debug_summary.json", "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2)
            print(f"[DEBUG] Execution complete. Debug artifacts saved in: {dbg_path.resolve()}\n")

        return GenerationResult(
            image_rgb=final_composite.image_rgb,
            image_bgr=final_composite.image_bgr,
            dsl_text=dsl_text,
            scene_ir=scene_ir,
            layout_plan=final_layout_plan,
            verification=final_verification,
            execution_trace=trace,
        )

    @staticmethod
    def _draw_layout_wireframe(layout_plan: LayoutPlan, background: Optional[np.ndarray] = None) -> np.ndarray:
        """Render a diagnostic wireframe canvas showing ground, horizon, and object boxes."""
        w = layout_plan.canvas_width
        h = layout_plan.canvas_height
        wireframe = np.zeros((h, w, 3), dtype=np.uint8)
        if background is not None:
            wireframe = cv2.resize(background[:, :, :3], (w, h)).copy()
        else:
            wireframe[:] = [35, 35, 35]

        # Ground line (green)
        cv2.line(wireframe, (0, layout_plan.ground_y), (w, layout_plan.ground_y), (0, 220, 0), 2)
        cv2.putText(wireframe, f"Ground (y={layout_plan.ground_y})", (20, layout_plan.ground_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

        # Horizon line (cyan)
        cv2.line(wireframe, (0, layout_plan.horizon_y), (w, layout_plan.horizon_y), (255, 200, 0), 1)
        cv2.putText(wireframe, f"Horizon (y={layout_plan.horizon_y})", (20, layout_plan.horizon_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # Object bounding boxes (bright colors)
        box_colors = [(0, 255, 255), (255, 0, 255), (0, 165, 255), (255, 255, 0), (100, 255, 100)]
        for idx, (name, obj) in enumerate(layout_plan.objects.items()):
            color = box_colors[idx % len(box_colors)]
            x1, y1 = obj.x, obj.y
            x2, y2 = obj.x + obj.width, obj.y + obj.height
            cv2.rectangle(wireframe, (x1, y1), (x2, y2), color, 2)
            label = f"{name} (z={obj.z_index})"
            cv2.putText(wireframe, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(wireframe, (x1 + obj.width // 2, y1 + obj.height // 2), 4, color, -1)

        return wireframe
