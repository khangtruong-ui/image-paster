"""Scene graph, layout, depth, and constraint resolution package."""

from image_paster.scene.graph import SceneGraph, GraphNode, GraphEdge
from image_paster.scene.depth import DepthSolver, DepthResolutionError
from image_paster.scene.constraints import ConstraintSolver, ConstraintViolation
from image_paster.scene.layout import (
    SemanticLayoutSolver,
    LayoutPlan,
    ObjectLayout,
)

__all__ = [
    "SceneGraph",
    "GraphNode",
    "GraphEdge",
    "DepthSolver",
    "DepthResolutionError",
    "ConstraintSolver",
    "ConstraintViolation",
    "SemanticLayoutSolver",
    "LayoutPlan",
    "ObjectLayout",
]
