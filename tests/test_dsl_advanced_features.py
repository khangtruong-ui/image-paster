"""Unit tests for advanced DSL features: linspace, summon, struct, and adjective validation."""

import pytest
import numpy as np
from image_paster.dsl import parse_dsl
from image_paster.dsl.parser import SceneDSLParser, DSLSyntaxError
from image_paster.dsl.validator import DSLValidator, DSLValidationError
from image_paster.scene.layout import SemanticLayoutSolver
from image_paster.segmentation.base import SegmentationResult
from image_paster.pipeline.generator import SemanticImageGenerator


def test_linspace_parsing_and_ir_expansion():
    dsl = """
    scene LinspaceScene {
        objects {
            object flower = search("yellow blooming flower");
            object flowers = linspace(flower, 5) {
                depth = foreground;
                region = bottom;
                standing_on = ground;
            }
        }
        relations {
            flowers.standing_on(ground);
        }
    }
    """
    parser = SceneDSLParser()
    ast = parser.parse(dsl)
    assert "flowers" in ast.objects
    assert ast.objects["flowers"].linspace_call is not None
    assert ast.objects["flowers"].linspace_call.target == "flower"
    assert ast.objects["flowers"].linspace_call.count == 5

    ir = parse_dsl(dsl, validate=True)
    # Check that 5 instances flowers_1 to flowers_5 are generated
    for i in range(1, 6):
        inst_name = f"flowers_{i}"
        assert inst_name in ir.objects
        assert ir.objects[inst_name].copied_from == "flower"
        assert ir.objects[inst_name].linspace_group == "flowers"
        assert ir.objects[inst_name].linspace_index == i - 1
        assert ir.objects[inst_name].linspace_total == 5
        assert ir.objects[inst_name].depth == "foreground"
        assert ir.objects[inst_name].source.query == "yellow blooming flower"

    # Verify layout solver positions them in a row
    solver = SemanticLayoutSolver(canvas_width=1000, canvas_height=800)
    layout = solver.solve(ir)
    x_positions = [layout.objects[f"flowers_{i}"].x for i in range(1, 6)]
    # x positions should be strictly increasing left to right
    for i in range(len(x_positions) - 1):
        assert x_positions[i] < x_positions[i + 1]


def test_summon_parsing_and_ir_expansion():
    dsl = """
    scene SummonScene {
        objects {
            object candle = search("lit wax candle");
            object candles = summon(candle, 6) {
                depth = foreground;
                standing_on = ground;
            }
        }
        relations {
            candles.standing_on(ground);
        }
    }
    """
    ir = parse_dsl(dsl, validate=True)
    # Check that 6 instances are generated
    for i in range(1, 7):
        inst_name = f"candles_{i}"
        assert inst_name in ir.objects
        assert ir.objects[inst_name].copied_from == "candle"
        assert ir.objects[inst_name].summon_group == "candles"
        assert ir.objects[inst_name].summon_index == i - 1
        assert ir.objects[inst_name].summon_total == 6

    # Verify layout solver positions them in a circle
    solver = SemanticLayoutSolver(canvas_width=1000, canvas_height=1000)
    layout = solver.solve(ir)
    coords = [(layout.objects[f"candles_{i}"].center) for i in range(1, 7)]
    # Coordinates should not all share the same x or y
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    assert max(xs) - min(xs) > 200
    assert max(ys) - min(ys) > 50


def test_struct_function_and_block_syntax():
    # 1. Function style struct
    dsl_func = """
    scene StructFuncScene {
        objects {
            object man = search("man in suit");
            object flower = search("red rose flower");
            object man_with_flower = struct(man, flower);
        }
    }
    """
    ir_func = parse_dsl(dsl_func, validate=True)
    assert "man_with_flower" in ir_func.objects
    assert ir_func.objects["man_with_flower"].is_composite is True
    assert ir_func.objects["man_with_flower"].struct_info is not None
    assert ir_func.objects["man_with_flower"].struct_info.base == "man"
    assert ir_func.objects["man_with_flower"].struct_info.parts == ["flower"]

    # 2. Block style struct
    dsl_block = """
    scene StructBlockScene {
        struct KnightShield {
            base = knight;
            part = shield;
        };
        objects {
            object knight = search("medieval knight in armor");
            object shield = search("metal knight shield");
        }
    }
    """
    ir_block = parse_dsl(dsl_block, validate=True)
    assert "KnightShield" in ir_block.objects
    assert ir_block.objects["KnightShield"].is_composite is True
    assert ir_block.objects["KnightShield"].struct_info.base == "knight"
    assert ir_block.objects["KnightShield"].struct_info.parts == ["shield"]


def test_nested_struct_in_linspace():
    dsl = """
    scene NestedScene {
        objects {
            object man = search("a man standing");
            object red_rose = search("a red rose flower");
            object men_with_roses = linspace(struct(man, red_rose), 4) {
                depth = foreground;
                region = bottom;
                standing_on = ground;
            }
        }
        relations {
            men_with_roses.standing_on(ground);
        }
    }
    """
    ir = parse_dsl(dsl, validate=True)
    assert "men_with_roses" in ir.objects
    assert ir.objects["men_with_roses"].linspace_info is not None
    assert ir.objects["men_with_roses"].linspace_info.count == 4

    # The 4 instances should be generated
    for i in range(1, 5):
        inst_name = f"men_with_roses_{i}"
        assert inst_name in ir.objects
        assert ir.objects[inst_name].copied_from == "men_with_roses_template"

    # Standalone base and parts should have is_template=True since not explicitly placed
    assert ir.objects["man"].is_template is True
    assert ir.objects["red_rose"].is_template is True

    # Layout solver should only layout the 4 instances, NOT standalone man or rose
    solver = SemanticLayoutSolver()
    layout = solver.solve(ir)
    assert "man" not in layout.objects
    assert "red_rose" not in layout.objects
    for i in range(1, 5):
        assert f"men_with_roses_{i}" in layout.objects


def test_nested_struct_in_summon():
    dsl = """
    scene SummonStructScene {
        objects {
            object wizard = search("a wizard with robe");
            object staff = search("magic wooden staff");
            object circle_of_wizards = summon(struct(wizard, staff), 6) {
                depth = foreground;
            }
        }
    }
    """
    ir = parse_dsl(dsl, validate=True)
    assert "circle_of_wizards" in ir.objects
    for i in range(1, 7):
        assert f"circle_of_wizards_{i}" in ir.objects
        assert ir.objects[f"circle_of_wizards_{i}"].copied_from == "circle_of_wizards_template"


def test_solitary_adjective_validation():
    # Solitary adjective as search query
    dsl_bad_query = """
    scene BadAdjectiveScene {
        objects {
            object item = search("red");
        }
    }
    """
    parser = SceneDSLParser()
    ast_bad = parser.parse(dsl_bad_query)
    validator = DSLValidator()
    with pytest.raises(DSLValidationError) as exc:
        validator.validate(ast_bad)
    assert "solitary adjective" in str(exc.value)

    # Valid noun phrase with adjective should pass
    dsl_good_query = """
    scene GoodNounScene {
        objects {
            object red_car = search("a red sports car");
        }
    }
    """
    ast_good = parser.parse(dsl_good_query)
    validator.validate(ast_good)  # Should not raise


def test_struct_cutout_compositing():
    # Create synthetic base and part segmentations
    base_rgba = np.zeros((200, 100, 4), dtype=np.uint8)
    base_rgba[20:180, 20:80] = [100, 150, 200, 255]
    base_seg = SegmentationResult(
        object_name="man",
        original_image=base_rgba[:, :, :3],
        mask=(base_rgba[:, :, 3] > 0).astype(np.uint8) * 255,
        extracted_rgba=base_rgba,
        bbox=(20, 20, 60, 160),
        score=0.95,
        rejected=False,
    )

    part_rgba = np.zeros((80, 80, 4), dtype=np.uint8)
    part_rgba[10:70, 10:70] = [220, 30, 30, 255]  # red flower
    part_seg = SegmentationResult(
        object_name="flower",
        original_image=part_rgba[:, :, :3],
        mask=(part_rgba[:, :, 3] > 0).astype(np.uint8) * 255,
        extracted_rgba=part_rgba,
        bbox=(10, 10, 60, 60),
        score=0.92,
        rejected=False,
    )

    comp_seg = SemanticImageGenerator._composite_struct_cutout(base_seg, [part_seg])
    assert comp_seg.width >= base_seg.width
    assert comp_seg.height >= base_seg.height
    assert comp_seg.rejected is False
    # Check that red pixels from the flower exist in the composite
    red_pixels = np.any((comp_seg.extracted_rgba[:, :, 0] > 180) & (comp_seg.extracted_rgba[:, :, 3] > 100))
    assert bool(red_pixels) is True


def test_to_cpp_dsl_roundtrip_with_advanced_features():
    dsl = """scene AdvancedRoundtripScene {
    camera {
        viewpoint = eye_level;
        perspective = natural;
    }
    environment {
        type = "garden";
    }
    objects {
        object flower {
            source {
                search("yellow rose flower");
            }
            depth = foreground;
        }
        object row_flowers = linspace(flower, 5) {
            depth = foreground;
            region = bottom;
        }
    }
    operations {
        retrieve;
        compose;
    }
}"""
    ir1 = parse_dsl(dsl, validate=True)
    serialized = ir1.to_cpp_dsl()
    assert "linspace(flower, 5)" in serialized
    # Re-parse serialized DSL
    ir2 = parse_dsl(serialized, validate=True)
    assert "row_flowers" in ir2.objects
    assert ir2.objects["row_flowers"].linspace_info.count == 5
    for i in range(1, 6):
        assert f"row_flowers_{i}" in ir2.objects
