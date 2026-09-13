"""Scene compositor rendering depth-ordered layers with seamless blending."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple
import numpy as np
import cv2

from image_paster.dsl.ir import SceneIR
from image_paster.scene.layout import LayoutPlan, ObjectLayout
from image_paster.segmentation.base import SegmentationResult
from image_paster.rendering.transforms import transform_object
from image_paster.rendering.masks import compute_canvas_occlusions
from image_paster.rendering.blending import (
    alpha_composite,
    poisson_blend,
    adjust_lighting_and_color,
)


@dataclass
class CompositeResult:
    """Output of composite rendering pipeline."""
    image_bgr: np.ndarray
    image_rgb: np.ndarray
    object_canvas_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    occlusion_stats: Dict[str, Any] = field(default_factory=dict)


class SceneCompositor:
    """Composites segmented objects onto scene background using depth ordering and Poisson blending."""

    def __init__(self, default_blend_mode: str = "poisson"):
        self.default_blend_mode = default_blend_mode

    def render(
        self,
        scene: SceneIR,
        layout_plan: LayoutPlan,
        segmentations: Dict[str, SegmentationResult],
        background_image: Optional[np.ndarray] = None,
        blend_mode: Optional[str] = None,
    ) -> CompositeResult:
        """Render complete composite scene."""
        w = layout_plan.canvas_width
        h = layout_plan.canvas_height
        mode = blend_mode or self.default_blend_mode

        # 1. Prepare base background
        if background_image is not None:
            bg_bgr = cv2.resize(background_image[:, :, :3], (w, h))
        else:
            bg_bgr = self._create_environment_background(scene, layout_plan)

        # 2. Sort objects back-to-front (ascending z_index)
        sorted_objects = sorted(
            layout_plan.objects.values(),
            key=lambda obj: obj.z_index,
        )

        canvas_masks: Dict[str, np.ndarray] = {}

        # 3. Composite each object
        for obj_layout in sorted_objects:
            name = obj_layout.name
            if name not in segmentations:
                continue

            seg = segmentations[name]
            raw_rgba = seg.extracted_rgba

            # Apply geometric transformations
            transformed_rgba = transform_object(
                rgba=raw_rgba,
                target_width=obj_layout.width,
                target_height=obj_layout.height,
                rotation=obj_layout.rotation,
                flip_h=obj_layout.flip_h,
                flip_v=obj_layout.flip_v,
            )

            # Apply appearance lighting & temperature adjustments
            obj_ir = scene.objects.get(name)
            temp = scene.environment.lighting.temperature
            contrast = 1.0
            brightness = 0.0
            if obj_ir and obj_ir.appearance:
                if obj_ir.appearance.contrast is not None:
                    contrast = obj_ir.appearance.contrast
                if obj_ir.appearance.brightness is not None:
                    brightness = obj_ir.appearance.brightness

            transformed_bgr = cv2.cvtColor(transformed_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            adjusted_bgr = adjust_lighting_and_color(
                transformed_bgr,
                temperature=temp,
                brightness=brightness,
                contrast=contrast,
            )
            # Reattach alpha
            adjusted_rgb = cv2.cvtColor(adjusted_bgr, cv2.COLOR_BGR2RGB)
            final_rgba = np.dstack([adjusted_rgb, transformed_rgba[:, :, 3]])

            # Create full canvas mask for occlusion analysis
            obj_mask = np.zeros((h, w), dtype=np.uint8)
            ox = max(0, obj_layout.x)
            oy = max(0, obj_layout.y)
            fw = min(w - ox, obj_layout.width)
            fh = min(h - oy, obj_layout.height)
            if fw > 0 and fh > 0:
                obj_mask[oy : oy + fh, ox : ox + fw] = final_rgba[:fh, :fw, 3]
            canvas_masks[name] = obj_mask

            # Blend onto background
            final_bgr = cv2.cvtColor(final_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            final_rgba_bgr = np.dstack([final_bgr, final_rgba[:, :, 3]])

            if mode == "poisson":
                bg_bgr = poisson_blend(
                    background_bgr=bg_bgr,
                    foreground_rgba=final_rgba_bgr,
                    x=obj_layout.x,
                    y=obj_layout.y,
                    blend_mode="normal",
                )
            else:
                bg_bgr = alpha_composite(
                    background_bgr=bg_bgr,
                    foreground_rgba=final_rgba_bgr,
                    x=obj_layout.x,
                    y=obj_layout.y,
                    opacity=obj_layout.opacity,
                )

        # 4. Compute occlusion metrics
        occlusion_stats = compute_canvas_occlusions(
            object_masks=canvas_masks,
            z_indices={k: v.z_index for k, v in layout_plan.objects.items()},
        )

        image_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
        return CompositeResult(
            image_bgr=bg_bgr,
            image_rgb=image_rgb,
            object_canvas_masks=canvas_masks,
            occlusion_stats=occlusion_stats,
        )

    def _create_environment_background(self, scene: SceneIR, layout_plan: LayoutPlan) -> np.ndarray:
        """Synthesize environment background with sky, horizon, and ground plane."""
        w = layout_plan.canvas_width
        h = layout_plan.canvas_height
        ground_y = layout_plan.ground_y
        env_type = scene.environment.env_type.lower()

        bg = np.zeros((h, w, 3), dtype=np.uint8)

        # Sky / upper gradient
        if "space" in env_type:
            sky_top = np.array([15, 10, 15], dtype=np.float32)
            sky_bot = np.array([40, 25, 45], dtype=np.float32)
            ground_top = np.array([60, 60, 65], dtype=np.float32)
            ground_bot = np.array([30, 30, 35], dtype=np.float32)
        elif "desert" in env_type or "beach" in env_type:
            sky_top = np.array([240, 200, 140], dtype=np.float32)
            sky_bot = np.array([255, 235, 190], dtype=np.float32)
            ground_top = np.array([140, 195, 220], dtype=np.float32)
            ground_bot = np.array([110, 170, 200], dtype=np.float32)
        elif "room" in env_type or "studio" in env_type:
            sky_top = np.array([220, 220, 220], dtype=np.float32)
            sky_bot = np.array([240, 240, 240], dtype=np.float32)
            ground_top = np.array([160, 180, 200], dtype=np.float32)
            ground_bot = np.array([130, 150, 170], dtype=np.float32)
        else:  # Forest / natural
            sky_top = np.array([230, 180, 120], dtype=np.float32)  # soft blue sky
            sky_bot = np.array([245, 220, 180], dtype=np.float32)
            ground_top = np.array([60, 130, 70], dtype=np.float32)  # green grassy ground
            ground_bot = np.array([40, 95, 50], dtype=np.float32)

        # Interpolate sky (0 .. ground_y)
        for y in range(ground_y):
            t = y / max(1, ground_y)
            color = (1.0 - t) * sky_top + t * sky_bot
            bg[y, :] = color.astype(np.uint8)

        # Interpolate ground (ground_y .. h)
        ground_height = max(1, h - ground_y)
        for y in range(ground_y, h):
            t = (y - ground_y) / ground_height
            color = (1.0 - t) * ground_top + t * ground_bot
            bg[y, :] = color.astype(np.uint8)

        # Soft horizon blend line
        if ground_y > 2 and ground_y < h - 2:
            bg[ground_y - 2 : ground_y + 2] = cv2.GaussianBlur(
                bg[ground_y - 4 : ground_y + 4], (5, 5), 1.0
            )[2:6]

        return bg
