"""Scene Intermediate Representation (IR) and canonical C++ code generation."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

from image_paster.dsl.ast_nodes import (
    SceneNode,
    CameraNode,
    EnvironmentNode,
    LightingNode,
    ObjectNode,
    SourceReqsNode,
    AppearanceNode,
    TransformationNode,
    RelationNode,
    ConstraintNode,
    OperationNode,
)


@dataclass
class CameraIR:
    viewpoint: str = "eye_level"
    perspective: str = "natural"
    focus: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ast(cls, node: Optional[CameraNode]) -> CameraIR:
        if not node:
            return cls()
        props = dict(node.properties)
        viewpoint = str(props.pop("viewpoint", "eye_level"))
        perspective = str(props.pop("perspective", "natural"))
        focus = str(props.pop("focus", "")) or None
        return cls(viewpoint=viewpoint, perspective=perspective, focus=focus, properties=props)


@dataclass
class LightingIR:
    direction: str = "upper_left"
    intensity: str = "medium"
    temperature: str = "natural"
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ast(cls, node: Optional[LightingNode]) -> LightingIR:
        if not node:
            return cls()
        props = dict(node.properties)
        direction = str(props.pop("direction", "upper_left"))
        intensity = str(props.pop("intensity", "medium"))
        temperature = str(props.pop("temperature", "natural"))
        return cls(direction=direction, intensity=intensity, temperature=temperature, properties=props)


@dataclass
class EnvironmentIR:
    env_type: str = "natural"
    query: Optional[str] = None
    sky: Optional[str] = None
    ground: Optional[str] = None
    lighting: LightingIR = field(default_factory=LightingIR)
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ast(cls, node: Optional[EnvironmentNode]) -> EnvironmentIR:
        if not node:
            return cls()
        lighting = LightingIR.from_ast(node.lighting)
        return cls(
            env_type=node.env_type,
            query=getattr(node, "query", None),
            sky=node.sky,
            ground=node.ground,
            lighting=lighting,
            properties=dict(node.properties),
        )


@dataclass
class SourceReqsIR:
    query: Optional[str] = None
    viewpoint: Optional[str] = None
    isolated: Optional[str] = None  # 'preferred', 'required'
    full_body: Optional[str] = None
    resolution: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ast(cls, node: Optional[SourceReqsNode]) -> SourceReqsIR:
        if not node:
            return cls()
        return cls(
            query=node.query,
            viewpoint=node.viewpoint,
            isolated=node.isolated,
            full_body=node.full_body,
            resolution=node.resolution,
            properties=dict(node.properties),
        )


@dataclass
class AppearanceIR:
    color: Optional[str] = None
    lighting: Optional[str] = "inherit_scene"
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    saturation: Optional[float] = None
    opacity: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ast(cls, node: Optional[AppearanceNode]) -> AppearanceIR:
        if not node:
            return cls()
        return cls(
            color=node.color,
            lighting=node.lighting or "inherit_scene",
            brightness=node.brightness,
            contrast=node.contrast,
            saturation=node.saturation,
            opacity=node.opacity if node.opacity is not None else 1.0,
            properties=dict(node.properties),
        )


@dataclass
class TransformIR:
    scale: Union[str, float] = "medium"
    facing: Optional[str] = None
    rotation: float = 0.0
    flip: Optional[str] = None
    perspective: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ast(cls, node: Optional[TransformationNode]) -> TransformIR:
        if not node:
            return cls()
        return cls(
            scale=node.scale if node.scale is not None else "medium",
            facing=node.facing,
            rotation=node.rotation if node.rotation is not None else 0.0,
            flip=node.flip,
            perspective=node.perspective,
            properties=dict(node.properties),
        )


@dataclass
class ObjectIR:
    name: str
    depth: str = "midground"  # 'foreground', 'midground', 'background', 'distant'
    region: Optional[str] = None
    standing_on: Optional[str] = None
    facing: Optional[str] = None
    source: SourceReqsIR = field(default_factory=SourceReqsIR)
    appearance: AppearanceIR = field(default_factory=AppearanceIR)
    transformation: TransformIR = field(default_factory=TransformIR)
    lighting: Optional[LightingIR] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ast(cls, node: ObjectNode) -> ObjectIR:
        source = SourceReqsIR.from_ast(node.source)
        appearance = AppearanceIR.from_ast(node.appearance)
        transformation = TransformIR.from_ast(node.transformation)
        lighting = LightingIR.from_ast(node.lighting) if node.lighting else None

        # Prioritize explicit properties over child blocks if set
        facing = node.facing or transformation.facing

        return cls(
            name=node.name,
            depth=node.depth or "midground",
            region=node.region,
            standing_on=node.standing_on,
            facing=facing,
            source=source,
            appearance=appearance,
            transformation=transformation,
            lighting=lighting,
            properties=dict(node.properties),
        )


@dataclass
class RelationIR:
    subject: str
    relation: str
    target: str
    params: List[Any] = field(default_factory=list)

    @classmethod
    def from_ast(cls, node: RelationNode) -> RelationIR:
        return cls(
            subject=node.subject,
            relation=node.relation,
            target=node.target,
            params=list(node.params),
        )


@dataclass
class ConstraintIR:
    subject: str
    constraint: str
    target: Optional[str] = None
    params: List[Any] = field(default_factory=list)

    @classmethod
    def from_ast(cls, node: ConstraintNode) -> ConstraintIR:
        return cls(
            subject=node.subject,
            constraint=node.constraint,
            target=node.target,
            params=list(node.params),
        )


@dataclass
class OperationIR:
    name: str

    @classmethod
    def from_ast(cls, node: OperationNode) -> OperationIR:
        return cls(name=node.name)


@dataclass
class SceneIR:
    """Scene Intermediate Representation."""
    name: str = "Scene"
    camera: CameraIR = field(default_factory=CameraIR)
    environment: EnvironmentIR = field(default_factory=EnvironmentIR)
    objects: Dict[str, ObjectIR] = field(default_factory=dict)
    relations: List[RelationIR] = field(default_factory=list)
    constraints: List[ConstraintIR] = field(default_factory=list)
    operations: List[OperationIR] = field(default_factory=list)

    @classmethod
    def from_ast(cls, node: SceneNode) -> SceneIR:
        camera = CameraIR.from_ast(node.camera)
        environment = EnvironmentIR.from_ast(node.environment)
        objects = {name: ObjectIR.from_ast(obj) for name, obj in node.objects.items()}
        relations = [RelationIR.from_ast(r) for r in node.relations]
        constraints = [ConstraintIR.from_ast(c) for c in node.constraints]
        operations = [OperationIR.from_ast(o) for o in node.operations]

        if not operations:
            # Default pipeline operations if none explicitly specified
            operations = [
                OperationIR("retrieve"),
                OperationIR("segment"),
                OperationIR("solve_layout"),
                OperationIR("compose"),
                OperationIR("blend"),
                OperationIR("verify"),
            ]

        return cls(
            name=node.name,
            camera=camera,
            environment=environment,
            objects=objects,
            relations=relations,
            constraints=constraints,
            operations=operations,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_cpp_dsl(self) -> str:
        """Serialize SceneIR back to canonical C++ style Scene DSL."""
        lines: List[str] = []
        lines.append(f"// Scene definition generated from SceneIR")
        lines.append(f"scene {self.name} {{")

        # Camera
        lines.append("    camera {")
        lines.append(f"        viewpoint = {self.camera.viewpoint};")
        lines.append(f"        perspective = {self.camera.perspective};")
        if self.camera.focus:
            lines.append(f"        focus = {self.camera.focus};")
        for k, v in self.camera.properties.items():
            val_str = f'"{v}"' if isinstance(v, str) else str(v)
            lines.append(f"        {k} = {val_str};")
        lines.append("    }")
        lines.append("")

        # Environment
        lines.append("    environment {")
        if self.environment.query:
            lines.append(f'        search("{self.environment.query}");')
        lines.append(f'        type = "{self.environment.env_type}";')
        if self.environment.sky:
            lines.append(f'        sky = "{self.environment.sky}";')
        if self.environment.ground:
            lines.append(f'        ground = "{self.environment.ground}";')
        lines.append("        lighting {")
        lines.append(f"            direction = {self.environment.lighting.direction};")
        lines.append(f"            intensity = {self.environment.lighting.intensity};")
        lines.append(f"            temperature = {self.environment.lighting.temperature};")
        lines.append("        }")
        lines.append("    }")
        lines.append("")

        # Objects
        lines.append("    objects {")
        for name, obj in self.objects.items():
            lines.append(f"        object {name} {{")
            # Source
            if any([obj.source.query, obj.source.viewpoint, obj.source.isolated, obj.source.full_body, obj.source.resolution]):
                lines.append("            source {")
                if obj.source.query:
                    lines.append(f'                search("{obj.source.query}");')
                if obj.source.viewpoint:
                    lines.append(f"                viewpoint = {obj.source.viewpoint};")
                if obj.source.isolated:
                    lines.append(f"                isolated = {obj.source.isolated};")
                if obj.source.full_body:
                    lines.append(f"                full_body = {obj.source.full_body};")
                if obj.source.resolution:
                    lines.append(f"                resolution = {obj.source.resolution};")
                lines.append("            }")

            # Depth & region
            lines.append(f"            depth = {obj.depth};")
            if obj.region:
                lines.append(f"            region = {obj.region};")
            if obj.standing_on:
                lines.append(f"            standing_on = {obj.standing_on};")
            if obj.facing:
                lines.append(f"            facing = {obj.facing};")

            # Appearance
            if any([obj.appearance.color, obj.appearance.lighting != "inherit_scene", obj.appearance.brightness is not None]):
                lines.append("            appearance {")
                if obj.appearance.color:
                    lines.append(f'                color = "{obj.appearance.color}";')
                if obj.appearance.lighting:
                    lines.append(f"                lighting = {obj.appearance.lighting};")
                if obj.appearance.brightness is not None:
                    lines.append(f"                brightness = {obj.appearance.brightness};")
                lines.append("            }")

            # Transformation
            lines.append("            transformation {")
            scale_val = f'"{obj.transformation.scale}"' if isinstance(obj.transformation.scale, str) and not obj.transformation.scale.isalnum() else obj.transformation.scale
            lines.append(f"                scale = {scale_val};")
            if obj.transformation.facing:
                lines.append(f"                facing = {obj.transformation.facing};")
            if obj.transformation.rotation != 0.0:
                lines.append(f"                rotation = {obj.transformation.rotation};")
            lines.append("            }")

            lines.append("        }")
        lines.append("    }")
        lines.append("")

        # Relations
        if self.relations:
            lines.append("    relations {")
            for rel in self.relations:
                if rel.params:
                    params_str = ", ".join(f'"{p}"' if isinstance(p, str) else str(p) for p in rel.params)
                    lines.append(f"        {rel.subject}.{rel.relation}({rel.target}, {params_str});")
                else:
                    lines.append(f"        {rel.subject}.{rel.relation}({rel.target});")
            lines.append("    }")
            lines.append("")

        # Constraints
        if self.constraints:
            lines.append("    constraints {")
            for cstr in self.constraints:
                if cstr.target and cstr.params:
                    params_str = ", ".join(f'"{p}"' if isinstance(p, str) else str(p) for p in cstr.params)
                    lines.append(f"        {cstr.subject}.{cstr.constraint}({cstr.target}, {params_str});")
                elif cstr.target:
                    lines.append(f"        {cstr.subject}.{cstr.constraint}({cstr.target});")
                else:
                    lines.append(f"        {cstr.subject}.{cstr.constraint}();")
            lines.append("    }")
            lines.append("")

        # Operations
        lines.append("    operations {")
        for op in self.operations:
            lines.append(f"        {op.name};")
        lines.append("    }")

        lines.append("}")
        return "\n".join(lines)
