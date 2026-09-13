"""Unit tests for Scene DSL semantic validator."""

import pytest
from image_paster.dsl.parser import SceneDSLParser
from image_paster.dsl.validator import DSLValidator, DSLValidationError


def test_validator_unknown_object():
    dsl = """
    scene UnknownObjScene {
        objects {
            object cat { depth = foreground; }
        }
        relations {
            cat.behind(dog); // 'dog' not defined
        }
    }
    """
    parser = SceneDSLParser()
    ast = parser.parse(dsl)
    validator = DSLValidator()
    with pytest.raises(DSLValidationError) as exc:
        validator.validate(ast)
    assert "dog" in str(exc.value)


def test_validator_contradictory_relations():
    dsl = """
    scene ContradictScene {
        objects {
            object a { depth = foreground; }
            object b { depth = foreground; }
        }
        relations {
            a.left_of(b);
            a.right_of(b); // Contradiction!
        }
    }
    """
    parser = SceneDSLParser()
    ast = parser.parse(dsl)
    validator = DSLValidator()
    with pytest.raises(DSLValidationError) as exc:
        validator.validate(ast)
    assert "Contradictory relations" in str(exc.value)


def test_validator_depth_cycle():
    dsl = """
    scene CycleScene {
        objects {
            object a { depth = foreground; }
            object b { depth = foreground; }
            object c { depth = foreground; }
        }
        relations {
            a.behind(b);
            b.behind(c);
            c.behind(a); // Cycle!
        }
    }
    """
    parser = SceneDSLParser()
    ast = parser.parse(dsl)
    validator = DSLValidator()
    with pytest.raises(DSLValidationError) as exc:
        validator.validate(ast)
    assert "Cyclic depth relationship" in str(exc.value)


def test_validator_invalid_depth():
    dsl = """
    scene InvalidDepthScene {
        objects {
            object a { depth = outer_space; } // Invalid depth
        }
    }
    """
    parser = SceneDSLParser()
    ast = parser.parse(dsl)
    validator = DSLValidator()
    with pytest.raises(DSLValidationError) as exc:
        validator.validate(ast)
    assert "invalid depth" in str(exc.value)
