"""Depth ordering and perspective scale resolution."""

from __future__ import annotations
from typing import Dict, List, Tuple
from image_paster.dsl.ir import SceneIR


BASE_DEPTH_MAP = {
    "distant": 0.15,
    "background": 0.35,
    "midground": 0.65,
    "foreground": 0.90,
}


class DepthResolutionError(Exception):
    """Exception raised when depth relationships cannot be resolved."""
    pass


class DepthSolver:
    """Solves depth ordering, assigns z-indices, and computes perspective scaling."""

    def __init__(self, depth_margin: float = 0.08):
        self.depth_margin = depth_margin

    def solve(self, scene: SceneIR) -> Dict[str, Tuple[float, int]]:
        """Solve depth values and discrete z-indices for all scene objects.

        Returns:
            Dictionary mapping object_name -> (continuous_depth [0.0..1.0], discrete_z_index).
        """
        objects = list(scene.objects.keys())
        if not objects:
            return {}

        # 1. Initialize depth from semantic layer
        depths: Dict[str, float] = {}
        for name, obj in scene.objects.items():
            base = BASE_DEPTH_MAP.get(obj.depth.lower(), 0.65)
            depths[name] = base

        # 2. Collect relative depth inequalities: (lesser, greater) where lesser is behind greater
        inequalities: List[Tuple[str, str]] = []

        # From relations
        for rel in scene.relations:
            subj = rel.subject
            tgt = rel.target
            if subj in depths and tgt in depths:
                r_type = rel.relation.lower()
                if r_type in ("behind", "further_than"):
                    inequalities.append((subj, tgt))
                elif r_type in ("in_front_of", "closer_than"):
                    inequalities.append((tgt, subj))

        # From constraints
        for cstr in scene.constraints:
            subj = cstr.subject
            tgt = cstr.target
            if subj in depths and tgt and tgt in depths:
                c_type = cstr.constraint.lower()
                if c_type in ("must_occlude", "occludes"):
                    # subject occludes target -> target is behind subject
                    inequalities.append((tgt, subj))
                elif c_type in ("must_be_behind", "behind"):
                    inequalities.append((subj, tgt))
                elif c_type in ("must_be_in_front_of", "in_front_of"):
                    inequalities.append((tgt, subj))

        # 3. Iterative relaxation to satisfy all depth inequalities
        max_iters = 50
        for _ in range(max_iters):
            changed = False
            for behind_obj, front_obj in inequalities:
                if depths[behind_obj] >= depths[front_obj]:
                    # Adjust: push behind_obj back, and front_obj forward
                    avg = (depths[behind_obj] + depths[front_obj]) / 2.0
                    depths[behind_obj] = max(0.05, avg - self.depth_margin / 2.0)
                    depths[front_obj] = min(0.98, avg + self.depth_margin / 2.0)
                    changed = True
            if not changed:
                break

        # 4. Verify all inequalities are satisfied
        for behind_obj, front_obj in inequalities:
            if depths[behind_obj] >= depths[front_obj]:
                raise DepthResolutionError(
                    f"Cannot satisfy depth ordering: '{behind_obj}' ({depths[behind_obj]:.2f}) "
                    f"must be behind '{front_obj}' ({depths[front_obj]:.2f})"
                )

        # 5. Assign discrete z-indices based on sorted depth values
        sorted_objs = sorted(objects, key=lambda o: depths[o])
        result: Dict[str, Tuple[float, int]] = {}
        for z_idx, name in enumerate(sorted_objs):
            result[name] = (depths[name], z_idx)

        return result

    @staticmethod
    def compute_perspective_scale(depth_value: float, perspective_mode: str = "natural") -> float:
        """Compute visual scale multiplier based on depth (0.0 = furthest, 1.0 = closest)."""
        mode = perspective_mode.lower()
        if mode == "telephoto":
            # Telephoto compresses depth (less variation)
            return 0.75 + 0.25 * depth_value
        elif mode == "wide_angle":
            # Wide angle exaggerates foreground
            return 0.35 + 0.65 * (depth_value ** 1.5)
        else:  # natural
            return 0.50 + 0.50 * depth_value
