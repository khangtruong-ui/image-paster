"""Abstract Syntax Tree (AST) node definitions for C++ style Scene DSL."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class CameraNode(ASTNode):
    """Camera specification AST node."""
    properties: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)


@dataclass
class LightingNode(ASTNode):
    """Lighting specification AST node."""
    properties: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)


@dataclass
class EnvironmentNode(ASTNode):
    """Environment specification AST node."""
    env_type: str = "studio"
    query: Optional[str] = None
    sky: Optional[str] = None
    ground: Optional[str] = None
    lighting: Optional[LightingNode] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceReqsNode(ASTNode):
    """Source image retrieval requirements AST node."""
    query: Optional[str] = None
    viewpoint: Optional[str] = None
    isolated: Optional[str] = None  # e.g., 'preferred', 'required'
    full_body: Optional[str] = None
    resolution: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AppearanceNode(ASTNode):
    """Appearance attributes AST node."""
    color: Optional[str] = None
    lighting: Optional[str] = None  # e.g., 'inherit_scene', 'custom'
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    saturation: Optional[float] = None
    opacity: Optional[float] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformationNode(ASTNode):
    """Geometric transformation AST node."""
    scale: Optional[Union[str, float]] = None  # 'small', 'medium', 'large' or float
    facing: Optional[str] = None  # 'left', 'right', 'toward_camera'
    rotation: Optional[float] = None
    flip: Optional[str] = None
    perspective: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectNode(ASTNode):
    """Scene object definition AST node."""
    name: str = ""
    depth: Optional[str] = None  # 'foreground', 'midground', 'background', 'distant'
    region: Optional[str] = None  # 'left', 'right', 'center', 'bottom', 'top'
    standing_on: Optional[str] = None  # 'ground', 'table', etc.
    facing: Optional[str] = None
    copied_from: Optional[str] = None  # Object name to copy from
    source: Optional[SourceReqsNode] = None
    appearance: Optional[AppearanceNode] = None
    transformation: Optional[TransformationNode] = None
    lighting: Optional[LightingNode] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodCallNode(ASTNode):
    """Method invocation in a chained call (e.g., .scale(0.8))."""
    method: str = ""
    args: List[Any] = field(default_factory=list)


@dataclass
class ChainedCallNode(ASTNode):
    """Chained method call AST node (e.g., car2.scale(0.8).facing(right);)."""
    target: str = ""
    calls: List[MethodCallNode] = field(default_factory=list)


@dataclass
class EditBlockNode(ASTNode):
    """Nested editing block representing a list of image edit operations."""
    items: List[Any] = field(default_factory=list)


@dataclass
class RelationNode(ASTNode):
    """Spatial or structural relation AST node."""
    subject: str = ""
    relation: str = ""  # 'behind', 'in_front_of', 'left_of', 'right_of', 'near', etc.
    target: str = ""
    params: List[Any] = field(default_factory=list)


@dataclass
class ConstraintNode(ASTNode):
    """Declarative constraint AST node."""
    subject: str = ""
    constraint: str = ""  # 'must_touch', 'must_occlude', 'must_be_inside', etc.
    target: Optional[str] = None
    params: List[Any] = field(default_factory=list)


@dataclass
class OperationNode(ASTNode):
    """Pipeline execution operation AST node."""
    name: str = ""
    details: Optional[Any] = None


@dataclass
class SceneNode(ASTNode):
    """Root scene AST node."""
    name: str = "Scene"
    camera: Optional[CameraNode] = None
    environment: Optional[EnvironmentNode] = None
    objects: Dict[str, ObjectNode] = field(default_factory=dict)
    edits: List[Any] = field(default_factory=list)
    relations: List[RelationNode] = field(default_factory=list)
    constraints: List[ConstraintNode] = field(default_factory=list)
    operations: List[OperationNode] = field(default_factory=list)
