"""Unit tests for Scene IR and C++ code generator roundtrip."""

from image_paster.dsl import parse_dsl


def test_ir_roundtrip_and_serialization():
    dsl = """
    scene RoundtripScene {
        camera {
            viewpoint = eye_level;
            perspective = natural;
            focus = elephant;
        }
        environment {
            type = "forest";
            lighting {
                direction = upper_left;
                intensity = medium;
                temperature = warm;
            }
        }
        objects {
            object elephant {
                source {
                    viewpoint = side;
                    isolated = preferred;
                    full_body = required;
                }
                depth = foreground;
                region = center;
                appearance {
                    color = "gray";
                    lighting = inherit_scene;
                }
                transformation {
                    scale = large;
                    facing = right;
                }
            }
        }
        relations {
            elephant.standing_on(ground);
        }
        constraints {
            elephant.must_touch(ground);
        }
        operations {
            retrieve;
            compose;
        }
    }
    """
    ir1 = parse_dsl(dsl, validate=True)
    cpp_code = ir1.to_cpp_dsl()

    # Re-parse generated C++ DSL
    ir2 = parse_dsl(cpp_code, validate=True)

    assert ir2.name == ir1.name
    assert ir2.camera.viewpoint == ir1.camera.viewpoint
    assert ir2.environment.env_type == ir1.environment.env_type
    assert list(ir2.objects.keys()) == list(ir1.objects.keys())
    assert ir2.objects["elephant"].depth == "foreground"
    assert ir2.objects["elephant"].source.viewpoint == "side"
    assert len(ir2.relations) == len(ir1.relations)
    assert len(ir2.constraints) == len(ir1.constraints)
