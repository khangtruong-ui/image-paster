"""Unit tests for C++ style Scene DSL parser."""

import pytest
from image_paster.dsl.parser import SceneDSLParser, DSLSyntaxError


def test_parse_valid_cpp_dsl():
    dsl_text = """
    // C++ style scene
    scene TestScene {
        camera {
            viewpoint = eye_level;
            perspective = natural;
            focus = hero;
        }

        environment {
            type = "forest";
            sky = "blue";
            ground = "grassy";
            lighting {
                direction = upper_left;
                intensity = medium;
                temperature = warm;
            }
        }

        objects {
            object hero {
                source {
                    viewpoint = side;
                    full_body = required;
                    isolated = preferred;
                }
                depth = foreground;
                region = center;
                standing_on = ground;
                appearance {
                    color = "brown";
                    lighting = inherit_scene;
                }
                transformation {
                    scale = large;
                    facing = right;
                }
            }
        }

        relations {
            hero.standing_on(ground);
        }

        constraints {
            hero.must_touch(ground);
        }

        operations {
            retrieve;
            segment;
            solve_layout;
            compose;
            blend;
            verify;
        }
    }
    """
    parser = SceneDSLParser()
    ast = parser.parse(dsl_text)
    assert ast.name == "TestScene"
    assert ast.camera is not None
    assert ast.camera.get("viewpoint") == "eye_level"
    assert ast.environment.env_type == "forest"
    assert "hero" in ast.objects
    assert ast.objects["hero"].source.viewpoint == "side"
    assert len(ast.relations) == 1
    assert ast.relations[0].subject == "hero"
    assert ast.relations[0].relation == "standing_on"
    assert ast.relations[0].target == "ground"
    assert len(ast.constraints) == 1
    assert len(ast.operations) == 6


def test_parse_syntax_error():
    invalid_dsl = """
    scene InvalidScene {
        camera {
            viewpoint = eye_level // missing semicolon
        }
    }
    """
    parser = SceneDSLParser()
    with pytest.raises(DSLSyntaxError) as exc_info:
        parser.parse(invalid_dsl)
    assert "DSLSyntaxError" in str(exc_info.value)


def test_parse_function_style_relations_and_comments():
    dsl = """
    /* Multi-line
       C++ style comment */
    scene AltScene {
        objects {
            object a { depth = foreground; }
            object b { depth = background; }
        }
        relations {
            behind(a, b);
            left_of(a, b);
        }
        constraints {
            must_touch(a, ground);
        }
        operations {
            compose;
        }
    }
    """
    parser = SceneDSLParser()
    ast = parser.parse(dsl)
    assert ast.name == "AltScene"
    assert len(ast.relations) == 2
    assert ast.relations[0].relation == "behind"
    assert ast.relations[0].subject == "a"
    assert ast.relations[0].target == "b"
    assert ast.constraints[0].constraint == "must_touch"
