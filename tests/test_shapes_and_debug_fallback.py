"""Unit tests for shapes rendering, shapes DSL, debug fallback detection, and raw reasoning logging."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from image_paster.dsl import parse_dsl, SceneIR, ShapeIR, ShapeNode, DSLValidationError
from image_paster.rendering.shapes import (
    render_circle,
    render_rectangle,
    render_triangle,
    render_line,
    render_curve,
    render_text,
    render_shape_cutout,
    parse_color,
)
from image_paster.llm.planner import (
    RuleBasedPlanner,
    TransformersPlanner,
    LLMScenePlanner,
    create_llm_planner,
)
from image_paster.pipeline.generator import SemanticImageGenerator
from image_paster.retrieval.mock import MockRetriever
from image_paster.segmentation.sam3 import SAM3Segmenter


def test_parse_color():
    # Hex colors
    assert parse_color("#FF0000") == (255, 0, 0, 255)
    assert parse_color("#00FF0080") == (0, 255, 0, 128)
    assert parse_color("#FFF") == (255, 255, 255, 255)

    # Named colors
    assert parse_color("red") == (255, 0, 0, 255)
    assert parse_color("white") == (255, 255, 255, 255)
    assert parse_color("black") == (0, 0, 0, 255)

    # RGB / RGBA strings
    assert parse_color("rgb(100, 150, 200)") == (100, 150, 200, 255)
    assert parse_color("rgba(100, 150, 200, 0.5)") == (100, 150, 200, 127)

    # Tuple
    assert parse_color((10, 20, 30)) == (10, 20, 30, 255)
    assert parse_color((10, 20, 30, 100)) == (10, 20, 30, 100)


def test_render_shapes_individual():
    # Circle
    circle_seg = render_circle(radius=40, color="yellow", stroke_color="orange", stroke_width=2)
    assert circle_seg.width == 80
    assert circle_seg.height == 80
    assert circle_seg.extracted_rgba.shape == (80, 80, 4)
    assert circle_seg.mask.shape == (80, 80)
    assert np.any(circle_seg.mask > 0)

    # Rectangle
    rect_seg = render_rectangle(width=100, height=50, color="blue", corner_radius=10)
    assert rect_seg.width == 100
    assert rect_seg.height == 50
    assert rect_seg.extracted_rgba.shape == (50, 100, 4)
    assert np.any(rect_seg.mask > 0)

    # Triangle
    tri_seg = render_triangle(base=80, height=60, color="green")
    assert tri_seg.width == 80
    assert tri_seg.height == 60
    assert np.any(tri_seg.mask > 0)

    # Line
    line_seg = render_line(x1=0, y1=0, x2=100, y2=100, stroke_width=4, color="red")
    assert line_seg.width >= 100
    assert line_seg.height >= 100
    assert np.any(line_seg.mask > 0)

    # Curve
    curve_seg = render_curve(points=[[0, 0], [50, 80], [100, 20]], stroke_width=3, color="purple")
    assert curve_seg.width >= 100
    assert np.any(curve_seg.mask > 0)

    # Text
    text_seg = render_text(content="Hello World", font_size=28, color="white", bg_color="rgba(0,0,0,0.5)")
    assert text_seg.width > 20
    assert text_seg.height > 10
    assert np.any(text_seg.mask > 0)


def test_render_shape_cutout_dispatch():
    shape_ir = ShapeIR(
        shape_type="circle",
        name="sun",
        radius=50,
        color="#FFD700",
        properties={"radius": 50, "color": "#FFD700", "blur": 2},
    )
    seg = render_shape_cutout(shape_ir)
    assert seg.object_name == "sun"
    assert seg.width == 100
    assert seg.height == 100
    assert not seg.rejected


def test_shapes_dsl_parsing_and_ir_roundtrip():
    dsl_text = """
    scene ShapeScene {
        environment {
            type = "desert";
        }
        shapes {
            circle sun {
                radius = 45;
                color = "#FFA500";
                region = top_right;
            }
            text title {
                content = "Desert Mirage";
                font_size = 32;
                color = "white";
                region = top_center;
            }
            rectangle sign_board {
                width = 250;
                height = 60;
                color = "#8B4513";
                region = bottom_left;
            }
            triangle pyramid {
                base = 180;
                height = 140;
                color = "#EDBB99";
                region = bottom_right;
            }
            line horizon {
                x1 = 0;
                y1 = 300;
                x2 = 800;
                y2 = 300;
                stroke_width = 2;
                color = "#F39C12";
            }
        }
        objects {
            object camel {
                source {
                    search("camel walking full body");
                }
                depth = foreground;
                region = center;
            }
        }
    }
    """
    ir = parse_dsl(dsl_text, validate=True)
    assert len(ir.shapes) == 5
    assert "sun" in ir.shapes
    assert ir.shapes["sun"].shape_type == "circle"
    assert ir.shapes["sun"].radius == 45
    assert "title" in ir.shapes
    assert ir.shapes["title"].shape_type == "text"
    assert ir.shapes["title"].text == "Desert Mirage"
    assert "camel" in ir.objects
    assert len(ir.objects) == 6  # 5 shapes + camel

    # Check round-trip serialization
    serialized = ir.to_cpp_dsl()
    assert "circle sun" in serialized
    assert "text title" in serialized
    assert "rectangle sign_board" in serialized
    assert "triangle pyramid" in serialized

    ir2 = parse_dsl(serialized, validate=True)
    assert len(ir2.shapes) == 5
    assert len(ir2.objects) == 6


def test_shapes_validation_errors():
    # Invalid shape type
    bad_type_dsl = """
    scene BadScene {
        shapes {
            pentagon star {
                color = "red";
            }
        }
    }
    """
    with pytest.raises(Exception):
        parse_dsl(bad_type_dsl, validate=True)

    # Negative dimension
    neg_dim_dsl = """
    scene BadScene {
        shapes {
            circle sun {
                radius = -20;
                color = "yellow";
            }
        }
    }
    """
    with pytest.raises(DSLValidationError):
        parse_dsl(neg_dim_dsl, validate=True)


def test_transformers_planner_raises_in_debug_on_fallback():
    # When debug=True and candidate model fails to load, must raise RuntimeError
    planner = TransformersPlanner(
        model_name="nonexistent_org/nonexistent_model_12345",
        debug=True,
    )
    with pytest.raises(RuntimeError) as excinfo:
        planner.plan("a cat on a sofa")
    assert "Model fallback occurred in debug mode" in str(excinfo.value)


def test_transformers_planner_graceful_when_not_debug():
    # When debug=False, fallback should occur gracefully without raising
    planner = TransformersPlanner(
        model_name="nonexistent_org/nonexistent_model_12345",
        debug=False,
    )
    with patch.object(planner, "_load_model", side_effect=RuntimeError("Simulated load failure")):
        dsl_text, ir = planner.plan("a cat on a sofa")
        assert planner.fallback_occurred is True
        assert ir is not None
        assert "cat" in ir.objects


def test_generator_raises_in_debug_when_planner_fell_back():
    mock_planner = MagicMock()
    mock_planner.debug = True
    mock_planner.fallback_occurred = True
    mock_planner.fallback_reason = "Simulated OOM fallback to secondary model"
    mock_planner.plan.return_value = (
        'scene Test { objects { object cat { source { search("cat"); } depth = foreground; } } }',
        parse_dsl('scene Test { objects { object cat { source { search("cat"); } depth = foreground; } } }', validate=False)
    )

    gen = SemanticImageGenerator(
        planner=mock_planner,
        retriever=MockRetriever(),
        debug=True,
    )
    with pytest.raises(RuntimeError) as excinfo:
        gen.generate("a cat", debug=True)
    assert "Model fallback occurred in debug mode" in str(excinfo.value)


def test_raw_llm_reasoning_captured_in_debug():
    with tempfile.TemporaryDirectory() as tmp_dir:
        planner = RuleBasedPlanner(creative=True, debug=True)
        gen = SemanticImageGenerator(
            planner=planner,
            retriever=MockRetriever(),
            segmenter=SAM3Segmenter(force_fallback=True),
            debug=True,
        )
        res = gen.generate("a peaceful forest glade with towering trees", debug=True, debug_dir=tmp_dir)

        assert res.execution_trace.get("raw_llm_response") is not None
        assert "Chain of Thought:" in res.execution_trace["raw_llm_response"]
        reasoning_file = Path(tmp_dir) / "00_llm_raw_reasoning.txt"
        assert reasoning_file.exists()
        content = reasoning_file.read_text(encoding="utf-8")
        assert len(content) > 0


def test_dense_scene_population_creative_vs_prompt_only():
    prompt = "a campsite in a pine forest"

    # Creative mode: should populate multiple rich contextual items
    planner_creative = RuleBasedPlanner(creative=True)
    _, ir_creative = planner_creative.plan(prompt)
    assert len(ir_creative.objects) >= 3  # subject + multiple creative additions

    # Prompt-only mode: only the explicit prompt objects
    planner_prompt_only = RuleBasedPlanner(creative=False)
    _, ir_prompt_only = planner_prompt_only.plan(prompt)
    assert len(ir_prompt_only.objects) <= 2


def test_pipeline_with_shapes_end_to_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dsl_with_shapes = """
        scene CanvasWithShapes {
            environment {
                type = "beach";
            }
            shapes {
                circle sun {
                    radius = 40;
                    color = "#F39C12";
                    region = top_right;
                }
                text banner {
                    content = "Summer Holiday";
                    font_size = 28;
                    color = "white";
                    region = top_center;
                }
                rectangle base_strip {
                    width = 300;
                    height = 40;
                    color = "rgba(0, 0, 0, 0.4)";
                    region = bottom_center;
                }
            }
            objects {
                object starfish {
                    source {
                        search("orange starfish on sand");
                    }
                    depth = foreground;
                    region = bottom_right;
                }
            }
        }
        """
        gen = SemanticImageGenerator(
            retriever=MockRetriever(),
            segmenter=SAM3Segmenter(force_fallback=True),
            debug=True,
        )
        res = gen.generate(
            prompt="shapes demo",
            dsl_override=dsl_with_shapes,
            debug=True,
            debug_dir=tmp_dir,
        )
        assert res.image_rgb is not None
        assert res.image_rgb.shape[0] > 0
        assert res.image_rgb.shape[1] > 0

        # Check debug cutouts were saved for shapes
        assert (Path(tmp_dir) / "02_segmentation_sun_cutout.png").exists()
        assert (Path(tmp_dir) / "02_segmentation_banner_cutout.png").exists()
        assert (Path(tmp_dir) / "02_segmentation_base_strip_cutout.png").exists()
