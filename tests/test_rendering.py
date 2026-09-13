"""Unit tests for geometric transformations, mask operations, blending, and compositor."""

import numpy as np
import cv2
from image_paster.rendering.transforms import transform_object
from image_paster.rendering.masks import clean_mask, feather_mask, compute_canvas_occlusions
from image_paster.rendering.blending import (
    alpha_composite,
    poisson_blend,
    adjust_lighting_and_color,
)
from image_paster.rendering.compositor import SceneCompositor
from image_paster.dsl import parse_dsl
from image_paster.scene.layout import SemanticLayoutSolver
from image_paster.segmentation.base import SegmentationResult


def test_transform_object():
    # 50x100 RGBA
    rgba = np.zeros((100, 50, 4), dtype=np.uint8)
    rgba[:, :, :3] = 120
    rgba[:, :, 3] = 255

    transformed = transform_object(
        rgba,
        target_width=200,
        target_height=300,
        rotation=90.0,
        flip_h=True,
    )
    assert transformed.shape == (300, 200, 4)


def test_mask_occlusion_calculation():
    h, w = 200, 200
    # Back object
    mask_a = np.zeros((h, w), dtype=np.uint8)
    mask_a[50:150, 50:150] = 255  # 100x100 = 10000 pixels

    # Front object overlapping right half of A
    mask_b = np.zeros((h, w), dtype=np.uint8)
    mask_b[50:150, 100:180] = 255  # overlaps 50x100 = 5000 pixels of A

    z_indices = {"a": 0, "b": 1}
    stats = compute_canvas_occlusions({"a": mask_a, "b": mask_b}, z_indices)

    assert stats["a"]["total_pixels"] == 10000
    assert stats["a"]["occluded_pixels"] == 5000
    assert abs(stats["a"]["visibility_ratio"] - 0.50) < 0.01
    assert stats["a"]["occluded_by"]["b"] == 5000
    assert stats["b"]["visibility_ratio"] == 1.0


def test_poisson_blend_and_boundary_fallback():
    bg = np.zeros((300, 300, 3), dtype=np.uint8)
    fg = np.ones((80, 80, 4), dtype=np.uint8) * 200

    # Normal inside placement
    res1 = poisson_blend(bg, fg, x=50, y=50, blend_mode="normal")
    assert res1.shape == (300, 300, 3)

    # Border placement (should fall back safely without crashing)
    res2 = poisson_blend(bg, fg, x=0, y=0)
    assert res2.shape == (300, 300, 3)


def test_lighting_temperature_adjustment():
    bgr = np.ones((50, 50, 3), dtype=np.uint8) * 128
    warm = adjust_lighting_and_color(bgr, temperature="warm")
    cool = adjust_lighting_and_color(bgr, temperature="cool")

    # Warm has higher Red (index 2 in BGR) than cool
    assert warm[0, 0, 2] > cool[0, 0, 2]
    # Cool has higher Blue (index 0 in BGR) than warm
    assert cool[0, 0, 0] > warm[0, 0, 0]


def test_compositor_render():
    dsl = """
    scene CompTest {
        objects {
            object ball {
                depth = foreground;
                region = center;
            }
        }
    }
    """
    ir = parse_dsl(dsl)
    solver = SemanticLayoutSolver(canvas_width=400, canvas_height=400)
    layout = solver.solve(ir)

    dummy_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    dummy_rgba[:, :, :3] = 200
    dummy_rgba[:, :, 3] = 255
    seg_res = SegmentationResult(
        object_name="ball",
        original_image=dummy_rgba[:, :, :3],
        mask=dummy_rgba[:, :, 3],
        extracted_rgba=dummy_rgba,
        bbox=(0, 0, 100, 100),
    )

    compositor = SceneCompositor()
    comp_res = compositor.render(ir, layout, {"ball": seg_res})
    assert comp_res.image_rgb.shape == (400, 400, 3)
    assert "ball" in comp_res.object_canvas_masks
