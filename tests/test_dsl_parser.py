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


def test_parse_search_call_variants():
    """Verify various forms of search call syntax parse query correctly."""
    parser = SceneDSLParser()

    # 1. Shorthand: name = search("...");
    dsl1 = """
    scene S1 {
        objects {
            elephant = search("red africa elephant");
        }
    }
    """
    ast1 = parser.parse(dsl1)
    assert "elephant" in ast1.objects
    assert ast1.objects["elephant"].source.query == "red africa elephant"

    # 2. Object with assignment to search: object name = search("...") { ... };
    dsl2 = """
    scene S2 {
        objects {
            object elephant = search("red africa elephant") {
                depth = foreground;
                region = center;
            }
        }
    }
    """
    ast2 = parser.parse(dsl2)
    assert ast2.objects["elephant"].source.query == "red africa elephant"
    assert ast2.objects["elephant"].depth == "foreground"

    # 3. Object body with search statement: search("...");
    dsl3 = """
    scene S3 {
        objects {
            object elephant {
                search("red africa elephant");
                depth = midground;
            }
        }
    }
    """
    ast3 = parser.parse(dsl3)
    assert ast3.objects["elephant"].source.query == "red africa elephant"

    # 4. Source block with search statement: source { search("..."); }
    dsl4 = """
    scene S4 {
        objects {
            object elephant {
                source {
                    search("red africa elephant");
                    viewpoint = side;
                }
            }
        }
    }
    """
    ast4 = parser.parse(dsl4)
    assert ast4.objects["elephant"].source.query == "red africa elephant"
    assert ast4.objects["elephant"].source.viewpoint == "side"

    # 5. Source block with query = "...": source { query = "..."; }
    dsl5 = """
    scene S5 {
        objects {
            object elephant {
                source {
                    query = "red africa elephant";
                }
            }
        }
    }
    """
    ast5 = parser.parse(dsl5)
    assert ast5.objects["elephant"].source.query == "red africa elephant"


def test_parse_environment_search():
    parser = SceneDSLParser()

    # 1. search("...") direct statement
    dsl1 = """
    scene EnvScene1 {
        environment {
            search("lush misty pine forest landscape photo");
            type = "forest";
            ground = "grassy";
        }
        objects {
            object tree { depth = foreground; }
        }
    }
    """
    ast1 = parser.parse(dsl1)
    assert ast1.environment.query == "lush misty pine forest landscape photo"
    assert ast1.environment.env_type == "forest"

    # 2. query = search("...") assignment
    dsl2 = """
    scene EnvScene2 {
        environment {
            query = search("sunny tropical beach ocean");
            type = "beach";
        }
        objects {
            object chair { depth = foreground; }
        }
    }
    """
    ast2 = parser.parse(dsl2)
    assert ast2.environment.query == "sunny tropical beach ocean"
    assert ast2.environment.env_type == "beach"

