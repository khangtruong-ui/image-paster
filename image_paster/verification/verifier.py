"""Visual and semantic scene verification."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import numpy as np

from image_paster.dsl.ir import SceneIR
from image_paster.scene.layout import LayoutPlan
from image_paster.rendering.compositor import CompositeResult


@dataclass
class VerificationResult:
    """Structured report produced by scene verification."""
    passed: bool
    score: float  # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_report(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"Verification Status: {status} (Score: {self.score:.2f})"]
        if self.issues:
            lines.append("Issues:")
            for issue in self.issues:
                lines.append(f"  - {issue}")
        if self.suggestions:
            lines.append("Suggestions for Retry:")
            for s in self.suggestions:
                lines.append(f"  - {s}")
        return "\n".join(lines)


class SceneVerifier(ABC):
    """Abstract interface for scene verifiers."""

    @abstractmethod
    def verify(
        self,
        scene: SceneIR,
        layout_plan: LayoutPlan,
        composite: CompositeResult,
    ) -> VerificationResult:
        """Evaluate whether composite scene satisfies semantic DSL."""
        pass


class SemanticVisualVerifier(SceneVerifier):
    """Deterministic visual & geometric verifier analyzing pixel masks, depth, and spatial relations."""

    def __init__(self, pass_threshold: float = 0.70):
        self.pass_threshold = pass_threshold

    def verify(
        self,
        scene: SceneIR,
        layout_plan: LayoutPlan,
        composite: CompositeResult,
    ) -> VerificationResult:
        issues: List[str] = []
        suggestions: List[str] = []
        checks_total = 0
        checks_passed = 0
        metrics: Dict[str, Any] = {}

        boxes = {k: v.bbox for k, v in layout_plan.objects.items()}
        z_indices = {k: v.z_index for k, v in layout_plan.objects.items()}
        ground_y = layout_plan.ground_y
        occlusion_stats = composite.occlusion_stats

        # 1. Verify Ground Contact
        for name, obj in scene.objects.items():
            if name not in layout_plan.objects:
                continue
            l = layout_plan.objects[name]
            checks_total += 1
            is_standing_ground = obj.standing_on == "ground" or any(
                r.subject == name and r.relation == "standing_on" and r.target == "ground"
                for r in scene.relations
            ) or any(
                c.subject == name and c.constraint in ("must_touch", "standing_on") and c.target == "ground"
                for c in scene.constraints
            )

            if is_standing_ground:
                bottom = l.y + l.height
                dist = abs(bottom - ground_y)
                if dist > 35:
                    issues.append(f"Object '{name}' does not touch ground (bottom={bottom}, ground={ground_y}, delta={dist}px)")
                    suggestions.append(f"Shift '{name}' vertically to align bottom with ground at y={ground_y}")
                else:
                    checks_passed += 1
            else:
                checks_passed += 1

        # 2. Verify Spatial Relations
        for rel in scene.relations:
            subj = rel.subject
            tgt = rel.target
            r_type = rel.relation.lower()

            if subj not in layout_plan.objects:
                continue

            sl = layout_plan.objects[subj]

            if tgt in layout_plan.objects:
                tl = layout_plan.objects[tgt]
                checks_total += 1

                if r_type == "behind":
                    # subj should have lower z_index than tgt
                    if sl.z_index >= tl.z_index:
                        issues.append(f"'{subj}' (z={sl.z_index}) is not behind '{tgt}' (z={tl.z_index})")
                        suggestions.append(f"Increase z_index of '{tgt}' relative to '{subj}'")
                    else:
                        checks_passed += 1

                elif r_type == "in_front_of":
                    if sl.z_index <= tl.z_index:
                        issues.append(f"'{subj}' (z={sl.z_index}) is not in front of '{tgt}' (z={tl.z_index})")
                        suggestions.append(f"Increase z_index of '{subj}' relative to '{tgt}'")
                    else:
                        checks_passed += 1

                elif r_type == "left_of":
                    if (sl.x + sl.width // 2) >= (tl.x + tl.width // 2):
                        issues.append(f"'{subj}' is not to the left of '{tgt}'")
                        suggestions.append(f"Translate '{subj}' leftward")
                    else:
                        checks_passed += 1

                elif r_type == "right_of":
                    if (sl.x + sl.width // 2) <= (tl.x + tl.width // 2):
                        issues.append(f"'{subj}' is not to the right of '{tgt}'")
                        suggestions.append(f"Translate '{subj}' rightward")
                    else:
                        checks_passed += 1

                elif r_type in ("standing_on", "sitting_on"):
                    bottom = sl.y + sl.height
                    target_top = tl.y
                    if abs(bottom - target_top) > 50:
                        issues.append(f"'{subj}' bottom ({bottom}) is not standing on '{tgt}' top ({target_top})")
                        suggestions.append(f"Align '{subj}' bottom with '{tgt}' top")
                    else:
                        checks_passed += 1

        # 3. Verify Occlusion Constraints
        for cstr in scene.constraints:
            subj = cstr.subject
            tgt = cstr.target
            c_type = cstr.constraint.lower()

            if subj not in layout_plan.objects:
                continue

            if c_type in ("must_occlude", "occludes") and tgt in layout_plan.objects:
                checks_total += 1
                # Subj should occlude tgt
                target_stats = occlusion_stats.get(tgt, {})
                occluded_by_subj = target_stats.get("occluded_by", {}).get(subj, 0)
                if occluded_by_subj <= 0:
                    issues.append(f"'{subj}' does not occlude '{tgt}' (0 overlapping pixels detected)")
                    suggestions.append(f"Reposition '{subj}' to partially overlap '{tgt}' horizontally")
                else:
                    checks_passed += 1

        # 4. Verify Visibility of Each Object
        for name in layout_plan.objects.keys():
            checks_total += 1
            stats = occlusion_stats.get(name, {})
            vis = stats.get("visibility_ratio", 1.0)
            metrics[f"{name}_visibility"] = vis
            if vis < 0.15:
                issues.append(f"Object '{name}' is almost completely occluded (visibility = {vis:.2f})")
                suggestions.append(f"Reduce overlap on '{name}' or bring it forward")
            else:
                checks_passed += 1

        score = checks_passed / max(1, checks_total)
        passed = (score >= self.pass_threshold) and (len(issues) == 0 or score > 0.85)

        return VerificationResult(
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions,
            metrics=metrics,
        )
