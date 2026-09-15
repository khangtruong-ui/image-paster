"""Scene DSL parsing, validation, and Intermediate Representation (IR)."""

from image_paster.dsl.ast_nodes import SceneNode, ShapeNode
from image_paster.dsl.parser import SceneDSLParser, DSLSyntaxError
from image_paster.dsl.validator import DSLValidator, DSLValidationError
from image_paster.dsl.ir import (
    SceneIR,
    CameraIR,
    EnvironmentIR,
    LightingIR,
    ObjectIR,
    SourceReqsIR,
    AppearanceIR,
    TransformIR,
    RelationIR,
    ConstraintIR,
    OperationIR,
    ShapeIR,
)


def parse_dsl(dsl_text: str, validate: bool = True) -> SceneIR:
    """Convenience function to parse and validate C++ style DSL text into SceneIR.

    Args:
        dsl_text: C++ style DSL string.
        validate: Whether to run semantic validation.

    Returns:
        SceneIR intermediate representation.
    """
    parser = SceneDSLParser()
    ast = parser.parse(dsl_text)
    if validate:
        validator = DSLValidator()
        validator.validate(ast)
    return SceneIR.from_ast(ast)


__all__ = [
    "SceneDSLParser",
    "DSLSyntaxError",
    "DSLValidator",
    "DSLValidationError",
    "SceneNode",
    "ShapeNode",
    "SceneIR",
    "CameraIR",
    "EnvironmentIR",
    "LightingIR",
    "ObjectIR",
    "SourceReqsIR",
    "AppearanceIR",
    "TransformIR",
    "RelationIR",
    "ConstraintIR",
    "OperationIR",
    "ShapeIR",
    "parse_dsl",
]
