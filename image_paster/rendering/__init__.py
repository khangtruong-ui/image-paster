"""Rendering and compositing package."""

from image_paster.rendering.transforms import transform_object
from image_paster.rendering.masks import (
    clean_mask,
    feather_mask,
    compute_canvas_occlusions,
)
from image_paster.rendering.blending import (
    alpha_composite,
    poisson_blend,
    adjust_lighting_and_color,
)
from image_paster.rendering.compositor import (
    SceneCompositor,
    CompositeResult,
)
from image_paster.rendering.shapes import (
    render_shape_cutout,
    render_circle,
    render_rectangle,
    render_triangle,
    render_line,
    render_curve,
    render_text,
    parse_color,
)

__all__ = [
    "transform_object",
    "clean_mask",
    "feather_mask",
    "compute_canvas_occlusions",
    "alpha_composite",
    "poisson_blend",
    "adjust_lighting_and_color",
    "SceneCompositor",
    "CompositeResult",
    "render_shape_cutout",
    "render_circle",
    "render_rectangle",
    "render_triangle",
    "render_line",
    "render_curve",
    "render_text",
    "parse_color",
]
