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
from image_paster.rendering.shapes import render_shape_cutout
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
        min_area_ratio: Optional[float] = None,
        max_area_ratio: Optional[float] = None,
    ):
        self.creative = creative
        self.planner = planner or create_llm_planner(creative=creative)
        # Fall back to mock retriever if DDG hits rate limits
        mock_retriever = MockRetriever()
        self.retriever = retriever or DuckDuckGoRetriever(fallback_retriever=mock_retriever)
        self.segmenter = segmenter or SAM3Segmenter()
        if min_area_ratio is not None and hasattr(self.segmenter, "min_area_ratio"):
            self.segmenter.min_area_ratio = min_area_ratio
        if max_area_ratio is not None and hasattr(self.segmenter, "max_area_ratio"):
            self.segmenter.max_area_ratio = max_area_ratio
        self.layout_solver = layout_solver or SemanticLayoutSolver()
        self.compositor = compositor or SceneCompositor(default_blend_mode="natural")
        self.verifier = verifier or SemanticVisualVerifier()
        self.max_retries = max_retries
        self.debug = debug
        if hasattr(self.planner, "debug") and self.debug:
            self.planner.debug = True

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
        if hasattr(self.planner, "debug"):
            self.planner.debug = is_debug

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
            "search_retry_history": [],
        }

        # 1. Scene Planning (LLM / Rule-Based) -> C++ DSL
        if dsl_override:
            dsl_text = dsl_override
            scene_ir = parse_dsl(dsl_text, validate=True)
            planner_type = "override"
        else:
            dsl_text, scene_ir = self.planner.plan(prompt)
            planner_type = self.planner.__class__.__name__

        # Check for model fallback in debug mode
        if is_debug and getattr(self.planner, "fallback_occurred", False):
            reason = getattr(self.planner, "fallback_reason", "Model fallback occurred from default model")
            raise RuntimeError(f"Model fallback occurred in debug mode: {reason}")

        raw_reasoning = getattr(self.planner, "last_raw_response", None)
        trace["planner"] = planner_type
        trace["dsl"] = dsl_text
        trace["scene_ir"] = scene_ir.to_dict()
        trace["chain_of_thought"] = scene_ir.chain_of_thought
        trace["raw_llm_response"] = raw_reasoning
        trace["llm_reasoning"] = raw_reasoning or scene_ir.chain_of_thought
        trace["pipeline_stages"].append("planning")

        if is_debug:
            print(f"\n[DEBUG:Planning] Planner: {planner_type}")
            print(f"[DEBUG:Planning] Compiled C++ Scene DSL:\n{dsl_text}\n")
            (dbg_path / "00_compiled_scene.dsl").write_text(dsl_text, encoding="utf-8")
            if raw_reasoning:
                print(f"[DEBUG:Planning] Full LLM Reasoning (Raw Text):\n{raw_reasoning}\n")
                (dbg_path / "00_llm_raw_reasoning.txt").write_text(raw_reasoning, encoding="utf-8")
            elif scene_ir.chain_of_thought:
                print(f"[DEBUG:Planning] Chain of Thought:\n{scene_ir.chain_of_thought}\n")
                (dbg_path / "00_llm_raw_reasoning.txt").write_text(scene_ir.chain_of_thought, encoding="utf-8")

        # Helper for background image retrieval
        def _fetch_background(env_ir, current_bg: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if current_bg is not None:
                return current_bg
            bg_query = env_ir.query
            if not bg_query and env_ir.env_type not in ("studio", "none"):
                bg_query = f"{env_ir.env_type} landscape background"

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
                            loaded_bg = np.array(bg_pil)
                            if is_debug:
                                dest_bg = dbg_path / "01_retrieval_background.png"
                                try:
                                    import shutil
                                    shutil.copyfile(img_path, dest_bg)
                                except Exception:
                                    pass
                            return loaded_bg
                        except Exception as e:
                            logger.warning(f"Failed to load background image candidate from {img_path}: {e}")
            return None

        # 2 & 3. Search & Segmentation Feedback Loop
        segmentations: Dict[str, SegmentationResult] = {}
        extracted_sizes: Dict[str, tuple[int, int]] = {}
        retrieval_results: Dict[str, RetrievalResult] = {}

        for search_attempt in range(self.max_retries + 1):
            segmentations.clear()
            extracted_sizes.clear()
            retrieval_results.clear()

            # Background retrieval
            background_image = _fetch_background(scene_ir.environment, background_image)

            # Separate shapes, copied, and non-copied objects
            shape_names = [name for name, obj in scene_ir.objects.items() if obj.shape_info is not None]
            copied_names = [name for name, obj in scene_ir.objects.items() if obj.copied_from and obj.shape_info is None]
            non_copied_names = [name for name, obj in scene_ir.objects.items() if not obj.copied_from and not obj.struct_info and obj.shape_info is None]

            # Directly render vector shapes & text cutouts
            for name in shape_names:
                obj = scene_ir.objects[name]
                shape_seg = render_shape_cutout(obj.shape_info, obj)
                segmentations[name] = shape_seg
                extracted_sizes[name] = (shape_seg.width, shape_seg.height)
                trace["segmentation"][name] = shape_seg.to_dict()
                if is_debug:
                    print(f"[DEBUG:Shapes] Rendered shape '{name}' (type={obj.shape_info.shape_type}): size={extracted_sizes[name]}")
                    shape_mask_file = dbg_path / f"02_segmentation_{name}_mask.png"
                    shape_cutout_file = dbg_path / f"02_segmentation_{name}_cutout.png"
                    cv2.imwrite(str(shape_mask_file), shape_seg.mask)
                    Image.fromarray(shape_seg.extracted_rgba).save(str(shape_cutout_file))

            # Parallel batch retrieval for all non-copied objects
            batch_reqs = [
                {
                    "object_name": name,
                    "source_reqs": scene_ir.objects[name].source,
                    "appearance": scene_ir.objects[name].appearance,
                    "max_results": 3,
                }
                for name in non_copied_names
            ]
            batch_results = self.retriever.retrieve_batch(batch_reqs)

            for name in non_copied_names:
                res = batch_results.get(name, RetrievalResult(object_name=name, query="", candidates=[]))
                retrieval_results[name] = res
                trace["retrieval"][name] = res.to_dict()
                if is_debug:
                    is_mock = any(c.image_url.startswith("mock://") for c in res.candidates)
                    source_label = "MOCK / SYNTHETIC" if is_mock else "REAL (DuckDuckGo)"
                    print(f"[DEBUG:Retrieval] Object '{name}': query='{res.query}', candidates={len(res.candidates)} [{source_label}]")
                    for cand in res.candidates:
                        print(f"      Candidate {cand.ranking}: url='{cand.image_url}'")
                        if cand.local_cached_path and Path(cand.local_cached_path).exists():
                            dest_cand = dbg_path / f"01_retrieval_{name}_{cand.ranking}.png"
                            try:
                                import shutil
                                shutil.copyfile(cand.local_cached_path, dest_cand)
                            except Exception:
                                pass

            for name in copied_names:
                parent = scene_ir.objects[name].copied_from
                res = RetrievalResult(
                    object_name=name,
                    query=f"copy({parent})",
                    candidates=[],
                )
                retrieval_results[name] = res
                trace["retrieval"][name] = res.to_dict()
                if is_debug:
                    print(f"[DEBUG:Retrieval] Object '{name}': copied from '{parent}' (skipping search retrieval)")

            if "retrieval" not in trace["pipeline_stages"]:
                trace["pipeline_stages"].append("retrieval")

            # Segmentation for non-copied objects
            if is_debug:
                active_m = getattr(self.segmenter, "active_model_name", None) or self.segmenter.__class__.__name__
                print(f"\n[DEBUG:Segmentation] Segmenting objects with {active_m} (Search Attempt {search_attempt + 1}):")

            search_failures: List[str] = []

            for name in non_copied_names:
                res = retrieval_results[name]
                obj_ir = scene_ir.objects.get(name)
                seg_prompt = obj_ir.source.query if (obj_ir and obj_ir.source.query) else name.replace("_", " ")

                if is_debug:
                    print(f"  --> Segmenting object: '{name}' (prompt='{seg_prompt}')")

                if not res.candidates:
                    search_failures.append(f"Object '{name}': 0 image candidates retrieved for query '{res.query}'")
                    continue

                seg_for_object = None
                chosen_candidate = None
                all_candidates_trace = []

                for cand in res.candidates:
                    img_path = cand.local_cached_path
                    if not img_path:
                        img_path = self.retriever.download_image(cand)
                    if not img_path:
                        continue

                    if is_debug and Path(img_path).exists():
                        dest_cand = dbg_path / f"01_retrieval_{name}_{cand.ranking}.png"
                        try:
                            import shutil
                            if not dest_cand.exists():
                                shutil.copyfile(img_path, dest_cand)
                        except Exception:
                            pass

                    # Run segmentation
                    seg_res = self.segmenter.segment(img_path, prompt=seg_prompt)
                    status_str = "ACCEPTED" if not seg_res.rejected else f"REJECTED ({seg_res.rejection_reason})"
                    status_label = "accepted" if not seg_res.rejected else "rejected"

                    if is_debug:
                        print(f"      Candidate {cand.ranking}: area={seg_res.area}px (ratio={seg_res.area_ratio:.3f}), score={seg_res.score:.2f}, bbox={seg_res.bbox} -> {status_str}")
                        cand_mask_file = dbg_path / f"02_segmentation_{name}_cand{cand.ranking}_{status_label}_mask.png"
                        cand_cutout_file = dbg_path / f"02_segmentation_{name}_cand{cand.ranking}_{status_label}_cutout.png"
                        cv2.imwrite(str(cand_mask_file), seg_res.mask)
                        Image.fromarray(seg_res.extracted_rgba).save(str(cand_cutout_file))

                    all_candidates_trace.append({
                        "ranking": cand.ranking,
                        "url": cand.image_url,
                        "area": seg_res.area,
                        "area_ratio": seg_res.area_ratio,
                        "score": float(seg_res.score),
                        "rejected": seg_res.rejected,
                        "rejection_reason": seg_res.rejection_reason,
                    })

                    if not seg_res.rejected and seg_for_object is None:
                        seg_for_object = seg_res
                        chosen_candidate = cand
                        break
                    elif seg_res.rejected:
                        logger.debug(f"Candidate {cand.ranking} for {name} rejected: {seg_res.rejection_reason}")

                if seg_for_object is None:
                    last_reason = all_candidates_trace[-1]["rejection_reason"] if all_candidates_trace else "No valid image candidates downloaded"
                    search_failures.append(f"Object '{name}': All {len(res.candidates)} candidates rejected by segmentation ({last_reason})")
                else:
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
                        "area_ratio": seg_for_object.area_ratio,
                        "width": seg_for_object.width,
                        "height": seg_for_object.height,
                        "all_candidates": all_candidates_trace,
                    }
                    if is_debug:
                        mask_file = dbg_path / f"02_segmentation_{name}_mask.png"
                        cutout_file = dbg_path / f"02_segmentation_{name}_cutout.png"
                        cv2.imwrite(str(mask_file), seg_for_object.mask)
                        Image.fromarray(seg_for_object.extracted_rgba).save(str(cutout_file))

            # Trigger search feedback loop if any failure and retries remaining
            if search_failures and search_attempt < self.max_retries:
                if is_debug:
                    print(f"\n[DEBUG:Retrieval Feedback] Failures encountered on attempt {search_attempt + 1}:")
                    for f_msg in search_failures:
                        print(f"  - {f_msg}")
                    print(f"[DEBUG:Retrieval Feedback] Replanning DSL with failure feedback...")
                trace["search_retry_history"].append({
                    "attempt": search_attempt + 1,
                    "failures": search_failures,
                })
                try:
                    new_dsl, new_scene_ir = self.planner.replan(
                        prompt=prompt,
                        previous_dsl=dsl_text,
                        failure_reasons=search_failures,
                    )
                    dsl_text = new_dsl
                    scene_ir = new_scene_ir
                    trace["dsl"] = dsl_text
                    trace["scene_ir"] = scene_ir.to_dict()
                    if is_debug:
                        print(f"[DEBUG:Planning] Replanned Scene DSL (Attempt {search_attempt + 2}):\n{dsl_text}\n")
                        (dbg_path / f"00_compiled_scene_retry_{search_attempt + 1}.dsl").write_text(dsl_text, encoding="utf-8")
                    continue
                except Exception as e:
                    logger.warning(f"Planner replan failed during search feedback: {e}")

            # Apply robust fallbacks for any remaining unsegmented non-copied objects
            if search_failures:
                for name in non_copied_names:
                    if name not in segmentations:
                        res = retrieval_results.get(name)
                        seg_prompt = scene_ir.objects[name].source.query or name
                        if is_debug:
                            print(f"      Warning: All candidates for '{name}' rejected; applying robust fallback segmentation.")
                        if res and res.candidates and res.candidates[0].local_cached_path:
                            fb_seg = self.segmenter.segment(res.candidates[0].local_cached_path, prompt=seg_prompt)
                            fb_seg.rejected = False
                            chosen_cand = res.candidates[0]
                        else:
                            dummy = np.zeros((400, 400, 4), dtype=np.uint8)
                            cv2.circle(dummy, (200, 200), 150, (180, 180, 180, 255), -1)
                            fb_seg = self.segmenter.segment(dummy, prompt=seg_prompt)
                            fb_seg.rejected = False
                            chosen_cand = None

                        segmentations[name] = fb_seg
                        extracted_sizes[name] = (fb_seg.width, fb_seg.height)
                        trace["segmentation"][name] = {
                            "object_name": name,
                            "segmentation_prompt": seg_prompt,
                            "segmenter": getattr(self.segmenter, "active_model_name", None) or self.segmenter.__class__.__name__,
                            "model_loaded": getattr(self.segmenter, "is_model_loaded", lambda: False)(),
                            "candidate_source": chosen_cand.source_url if chosen_cand else "synthetic_fallback",
                            "score": float(fb_seg.score),
                            "bbox": fb_seg.bbox,
                            "rejected": False,
                            "rejection_reason": "Force accepted after retries exhausted",
                            "area": fb_seg.area,
                            "area_ratio": fb_seg.area_ratio,
                            "width": fb_seg.width,
                            "height": fb_seg.height,
                        }
                        if is_debug:
                            mask_file = dbg_path / f"02_segmentation_{name}_mask.png"
                            cutout_file = dbg_path / f"02_segmentation_{name}_cutout.png"
                            cv2.imwrite(str(mask_file), fb_seg.mask)
                            Image.fromarray(fb_seg.extracted_rgba).save(str(cutout_file))

            # Resolve struct composite objects
            composite_names = [name for name, obj in scene_ir.objects.items() if obj.struct_info]
            for comp_name in composite_names:
                info = scene_ir.objects[comp_name].struct_info
                b_name = info.base
                p_names = info.parts
                b_seg = segmentations.get(b_name)
                p_segs = [segmentations[p] for p in p_names if p in segmentations]
                if b_seg:
                    comp_seg = self._composite_struct_cutout(b_seg, p_segs)
                    comp_seg.object_name = comp_name
                    segmentations[comp_name] = comp_seg
                    extracted_sizes[comp_name] = (comp_seg.width, comp_seg.height)
                    trace["segmentation"][comp_name] = {
                        "object_name": comp_name,
                        "struct_base": b_name,
                        "struct_parts": p_names,
                        "segmenter": "composite_struct",
                        "score": float(comp_seg.score),
                        "width": comp_seg.width,
                        "height": comp_seg.height,
                    }
                    if is_debug:
                        print(f"[DEBUG:Segmentation] Object '{comp_name}' successfully formed composite struct from base '{b_name}' and parts {p_names}")
                        mask_file = dbg_path / f"02_segmentation_{comp_name}_mask.png"
                        cutout_file = dbg_path / f"02_segmentation_{comp_name}_cutout.png"
                        cv2.imwrite(str(mask_file), comp_seg.mask)
                        Image.fromarray(comp_seg.extracted_rgba).save(str(cutout_file))

            # Resolve copied objects
            unresolved = set(copied_names)
            while unresolved:
                resolved_any = False
                for c_name in list(unresolved):
                    parent = scene_ir.objects[c_name].copied_from
                    if parent in segmentations:
                        p_seg = segmentations[parent]
                        c_seg = SegmentationResult(
                            object_name=c_name,
                            original_image=p_seg.original_image.copy(),
                            mask=p_seg.mask.copy(),
                            extracted_rgba=p_seg.extracted_rgba.copy(),
                            bbox=p_seg.bbox,
                            score=p_seg.score,
                            rejected=p_seg.rejected,
                            rejection_reason=p_seg.rejection_reason,
                        )
                        segmentations[c_name] = c_seg
                        extracted_sizes[c_name] = (c_seg.width, c_seg.height)
                        trace["segmentation"][c_name] = {
                            "object_name": c_name,
                            "copied_from": parent,
                            "segmentation_prompt": f"copy({parent})",
                            "segmenter": "copied",
                            "model_loaded": True,
                            "candidate_source": f"copied from {parent}",
                            "score": float(c_seg.score),
                            "bbox": c_seg.bbox,
                            "rejected": c_seg.rejected,
                            "rejection_reason": c_seg.rejection_reason,
                            "area": c_seg.area,
                            "area_ratio": c_seg.area_ratio,
                            "width": c_seg.width,
                            "height": c_seg.height,
                        }
                        if is_debug:
                            print(f"[DEBUG:Segmentation] Object '{c_name}' successfully copied segmentation from '{parent}'")
                            mask_file = dbg_path / f"02_segmentation_{c_name}_mask.png"
                            cutout_file = dbg_path / f"02_segmentation_{c_name}_cutout.png"
                            cv2.imwrite(str(mask_file), c_seg.mask)
                            Image.fromarray(c_seg.extracted_rgba).save(str(cutout_file))
                        unresolved.remove(c_name)
                        resolved_any = True
                if not resolved_any:
                    # Unresolvable cycle or parent missing
                    for c_name in unresolved:
                        dummy = np.zeros((400, 400, 4), dtype=np.uint8)
                        cv2.circle(dummy, (200, 200), 150, (180, 180, 180, 255), -1)
                        fb_seg = self.segmenter.segment(dummy, prompt=c_name)
                        fb_seg.rejected = False
                        segmentations[c_name] = fb_seg
                        extracted_sizes[c_name] = (fb_seg.width, fb_seg.height)
                    break

            # Search loop succeeded or fallbacks applied
            break

        if "segmentation" not in trace["pipeline_stages"]:
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

    @staticmethod
    def _composite_struct_cutout(base_seg: SegmentationResult, part_segs: List[SegmentationResult]) -> SegmentationResult:
        """Composite part segmentation cutouts onto base object cutout to create composite struct cutout."""
        base_rgba = base_seg.extracted_rgba.copy()
        bh, bw = base_rgba.shape[:2]

        curr_rgba = base_rgba
        for p_idx, p_seg in enumerate(part_segs):
            p_rgba = p_seg.extracted_rgba
            ph, pw = p_rgba.shape[:2]
            if ph == 0 or pw == 0:
                continue

            # Scale part to ~35% of base height, preserving aspect ratio
            target_ph = max(20, int(bh * 0.35))
            scale = target_ph / float(ph)
            target_pw = max(20, int(pw * scale))
            resized_part = cv2.resize(p_rgba, (target_pw, target_ph), interpolation=cv2.INTER_AREA)

            # Attachment point: right-hand/chest holding position on base
            offset_x = int(bw * 0.45) + p_idx * 15
            offset_y = int(bh * 0.40) + p_idx * 15

            # Canvas expansion if needed
            max_x = max(curr_rgba.shape[1], offset_x + target_pw)
            max_y = max(curr_rgba.shape[0], offset_y + target_ph)
            min_x = min(0, offset_x)
            min_y = min(0, offset_y)

            new_w = max_x - min_x
            new_h = max_y - min_y
            expanded = np.zeros((new_h, new_w, 4), dtype=np.uint8)

            # Place base in expanded canvas
            base_x = -min_x
            base_y = -min_y
            expanded[base_y : base_y + curr_rgba.shape[0], base_x : base_x + curr_rgba.shape[1]] = curr_rgba

            # Alpha composite resized_part
            px = offset_x - min_x
            py = offset_y - min_y
            part_rgb = resized_part[:, :, :3].astype(np.float32)
            part_alpha = (resized_part[:, :, 3].astype(np.float32) / 255.0)[:, :, None]

            target_roi = expanded[py : py + target_ph, px : px + target_pw]
            target_rgb = target_roi[:, :, :3].astype(np.float32)
            target_alpha = (target_roi[:, :, 3].astype(np.float32) / 255.0)[:, :, None]

            out_alpha = part_alpha + target_alpha * (1.0 - part_alpha)
            out_rgb = (part_rgb * part_alpha + target_rgb * target_alpha * (1.0 - part_alpha)) / np.maximum(1e-5, out_alpha)

            target_roi[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
            target_roi[:, :, 3] = np.clip(out_alpha[:, :, 0] * 255.0, 0, 255).astype(np.uint8)
            curr_rgba = expanded

        mask = (curr_rgba[:, :, 3] > 10).astype(np.uint8) * 255
        return SegmentationResult(
            object_name=base_seg.object_name,
            original_image=curr_rgba[:, :, :3].copy(),
            mask=mask,
            extracted_rgba=curr_rgba,
            bbox=(0, 0, curr_rgba.shape[1], curr_rgba.shape[0]),
            score=base_seg.score,
            rejected=False,
        )

