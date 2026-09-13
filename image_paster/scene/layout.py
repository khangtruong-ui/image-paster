"""Semantic layout solver converting semantic DSL relations into concrete canvas coordinates."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Optional, Any, List

from image_paster.dsl.ir import SceneIR
from image_paster.scene.depth import DepthSolver
from image_paster.scene.constraints import ConstraintSolver, ConstraintViolation


@dataclass
class ObjectLayout:
    """Concrete numerical rendering layout for an object."""
    name: str
    x: int
    y: int
    width: int
    height: int
    z_index: int
    depth_value: float
    scale: float
    rotation: float = 0.0
    flip_h: bool = False
    flip_v: bool = False
    opacity: float = 1.0
    anchor: Tuple[float, float] = (0.5, 1.0)  # (center_x, bottom_y) default anchor
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayoutPlan:
    """Complete layout plan for rendering a scene."""
    canvas_width: int
    canvas_height: int
    ground_y: int
    horizon_y: int
    objects: Dict[str, ObjectLayout] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "ground_y": self.ground_y,
            "horizon_y": self.horizon_y,
            "objects": {k: v.to_dict() for k, v in self.objects.items()},
        }


SCALE_MAP = {
    "tiny": 0.18,
    "small": 0.32,
    "medium": 0.50,
    "large": 0.72,
    "huge": 0.90,
}

REGION_X_MAP = {
    "left": 0.22,
    "top_left": 0.22,
    "bottom_left": 0.22,
    "center": 0.50,
    "right": 0.78,
    "top_right": 0.78,
    "bottom_right": 0.78,
}


class SemanticLayoutSolver:
    """Derives numerical coordinates and transformations from semantic scene specifications."""

    def __init__(
        self,
        canvas_width: int = 1024,
        canvas_height: int = 1024,
        depth_solver: Optional[DepthSolver] = None,
    ):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.depth_solver = depth_solver or DepthSolver()

    def solve(
        self,
        scene: SceneIR,
        extracted_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> LayoutPlan:
        """Solve layout for given scene and object sizes."""
        w, h = self.canvas_width, self.canvas_height

        # 1. Camera horizon & ground line
        vp = scene.camera.viewpoint.lower()
        if vp == "high_angle":
            horizon_y = int(h * 0.35)
            ground_y = int(h * 0.85)
        elif vp == "low_angle":
            horizon_y = int(h * 0.70)
            ground_y = int(h * 0.65)
        else:  # eye_level
            horizon_y = int(h * 0.50)
            ground_y = int(h * 0.75)

        # 2. Solve depth ordering
        depth_info = self.depth_solver.solve(scene)

        # 3. Base dimensions and scales
        obj_layouts: Dict[str, ObjectLayout] = {}
        aspect_ratios: Dict[str, float] = {}

        for name, obj in scene.objects.items():
            if extracted_sizes and name in extracted_sizes:
                orig_w, orig_h = extracted_sizes[name]
                ar = max(0.1, orig_w / max(1, orig_h))
            else:
                ar = 1.0
            aspect_ratios[name] = ar

            # Base scale
            raw_scale = obj.transformation.scale
            if isinstance(raw_scale, (int, float)):
                base_scale = float(raw_scale)
            else:
                base_scale = SCALE_MAP.get(str(raw_scale).lower(), 0.50)

            # Modulate by depth and perspective
            depth_val, z_idx = depth_info.get(name, (0.65, 0))
            persp_scale = DepthSolver.compute_perspective_scale(depth_val, scene.camera.perspective)
            eff_scale = base_scale * persp_scale

            # Target object height
            target_h = int(h * 0.60 * eff_scale)
            target_w = int(target_h * ar)
            # Clamp to canvas boundaries
            target_w = max(40, min(w - 20, target_w))
            target_h = max(40, min(h - 20, target_h))

            # Initial center coordinates
            cx = int(w * 0.5)
            if obj.region and obj.region.lower() in REGION_X_MAP:
                cx = int(w * REGION_X_MAP[obj.region.lower()])

            # Initial vertical positioning (standing on ground by default)
            cy = ground_y - (target_h // 2)

            # Flip logic
            flip_h = False
            facing = (obj.facing or obj.transformation.facing or "").lower()
            if facing == "left":
                flip_h = True

            obj_layouts[name] = ObjectLayout(
                name=name,
                x=cx - target_w // 2,
                y=cy - target_h // 2,
                width=target_w,
                height=target_h,
                z_index=z_idx,
                depth_value=depth_val,
                scale=eff_scale,
                rotation=obj.transformation.rotation,
                flip_h=flip_h,
                opacity=obj.appearance.opacity,
            )

        # 4. Resolve spatial relations
        for rel in scene.relations:
            subj = rel.subject
            tgt = rel.target
            r_type = rel.relation.lower()

            if subj not in obj_layouts:
                continue

            slayout = obj_layouts[subj]

            if tgt == "ground":
                # Pin bottom to ground
                slayout.y = ground_y - slayout.height
                continue

            if tgt in obj_layouts:
                tlayout = obj_layouts[tgt]

                if r_type == "left_of":
                    # Place subject to left of target
                    slayout.x = tlayout.x - slayout.width - int(w * 0.04)
                elif r_type == "right_of":
                    slayout.x = tlayout.x + tlayout.width + int(w * 0.04)
                elif r_type == "above":
                    slayout.y = tlayout.y - slayout.height - int(h * 0.02)
                elif r_type == "below":
                    slayout.y = tlayout.y + tlayout.height + int(h * 0.02)
                elif r_type in ("standing_on", "sitting_on"):
                    # Stand on top of target
                    slayout.x = tlayout.x + (tlayout.width - slayout.width) // 2
                    slayout.y = tlayout.y - int(slayout.height * 0.85)
                elif r_type == "near":
                    # Keep horizontally adjacent with small gap
                    if slayout.x <= tlayout.x:
                        slayout.x = tlayout.x - slayout.width - int(w * 0.02)
                    else:
                        slayout.x = tlayout.x + tlayout.width + int(w * 0.02)
                elif r_type == "behind":
                    # Position with substantial horizontal overlap so object masks intersect
                    overlap = int(min(slayout.width, tlayout.width) * 0.45)
                    # If subject is on right or center, place to overlap with target
                    if slayout.x <= tlayout.x:
                        slayout.x = tlayout.x - slayout.width + overlap
                    else:
                        slayout.x = tlayout.x + tlayout.width - overlap
                    # Slightly raise in perspective
                    slayout.y = tlayout.y + (tlayout.height - slayout.height) - int(h * 0.04)

        # 5. Handle ground contact alignment for objects standing on ground
        for name, obj in scene.objects.items():
            if obj.standing_on == "ground" or any(r.subject == name and r.relation == "standing_on" and r.target == "ground" for r in scene.relations):
                l = obj_layouts[name]
                l.y = ground_y - l.height

        # 6. Keep all objects inside canvas margins
        for l in obj_layouts.values():
            l.x = max(10, min(w - l.width - 10, l.x))
            l.y = max(10, min(h - l.height - 10, l.y))

        # 7. Check constraints and apply repairs
        violations = ConstraintSolver.evaluate(
            constraints=scene.constraints,
            boxes={k: v.bbox for k, v in obj_layouts.items()},
            z_indices={k: v.z_index for k, v in obj_layouts.items()},
            ground_y=ground_y,
            canvas_w=w,
            canvas_h=h,
        )

        # Attempt automatic resolution for minor violations
        if violations:
            for v in violations:
                if v.constraint in ("must_touch", "standing_on") and v.target == "ground":
                    # Fix bottom to ground
                    obj_layouts[v.subject].y = ground_y - obj_layouts[v.subject].height
                elif v.constraint in ("must_touch", "standing_on") and v.target in obj_layouts:
                    # Align bottom of subject with top of target
                    obj_layouts[v.subject].y = obj_layouts[v.target].y - obj_layouts[v.subject].height
                elif v.constraint in ("must_be_larger_than", "larger_than") and v.target in obj_layouts:
                    # Upscale subject to exceed target area
                    s_obj = obj_layouts[v.subject]
                    t_obj = obj_layouts[v.target]
                    ratio = ((t_obj.width * t_obj.height * 1.25) / max(1, s_obj.width * s_obj.height)) ** 0.5
                    s_obj.width = int(s_obj.width * ratio)
                    s_obj.height = int(s_obj.height * ratio)
                    # Re-align ground contact if applicable
                    s_ir = scene.objects.get(v.subject)
                    if s_ir and (s_ir.standing_on == "ground" or any(r.subject == v.subject and r.relation == "standing_on" and r.target == "ground" for r in scene.relations)):
                        s_obj.y = ground_y - s_obj.height
                    # Re-align objects standing on this subject
                    for rel in scene.relations:
                        if rel.relation in ("standing_on", "sitting_on") and rel.target == v.subject and rel.subject in obj_layouts:
                            obj_layouts[rel.subject].y = s_obj.y - obj_layouts[rel.subject].height
                elif v.constraint in ("must_be_smaller_than", "smaller_than") and v.target in obj_layouts:
                    # Downscale subject
                    s_obj = obj_layouts[v.subject]
                    t_obj = obj_layouts[v.target]
                    ratio = ((t_obj.width * t_obj.height * 0.75) / max(1, s_obj.width * s_obj.height)) ** 0.5
                    s_obj.width = int(s_obj.width * ratio)
                    s_obj.height = int(s_obj.height * ratio)
                    s_ir = scene.objects.get(v.subject)
                    if s_ir and (s_ir.standing_on == "ground" or any(r.subject == v.subject and r.relation == "standing_on" and r.target == "ground" for r in scene.relations)):
                        s_obj.y = ground_y - s_obj.height
                elif v.constraint in ("must_occlude", "occludes") and v.target in obj_layouts:
                    # Nudge subject to overlap target
                    occ = obj_layouts[v.subject]
                    target_b = obj_layouts[v.target]
                    occ.x = target_b.x + int(target_b.width * 0.2)

            # Re-evaluate
            violations = ConstraintSolver.evaluate(
                constraints=scene.constraints,
                boxes={k: v.bbox for k, v in obj_layouts.items()},
                z_indices={k: v.z_index for k, v in obj_layouts.items()},
                ground_y=ground_y,
                canvas_w=w,
                canvas_h=h,
            )

            if violations:
                # If still unresolved, raise the first violation as structured error
                raise violations[0]

        return LayoutPlan(
            canvas_width=w,
            canvas_height=h,
            ground_y=ground_y,
            horizon_y=horizon_y,
            objects=obj_layouts,
        )
