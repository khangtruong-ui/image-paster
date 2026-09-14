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
    ChainedCallNode,
    EditBlockNode,
    StructCallNode,
    LinspaceNode,
    SummonNode,
    StructBlockNode,
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
class StructIR:
    base: str
    parts: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinspaceIR:
    target: str
    count: int = 3
    struct_info: Optional[StructIR] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SummonIR:
    target: str
    count: int = 6
    struct_info: Optional[StructIR] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectIR:
    name: str
    depth: str = "midground"  # 'foreground', 'midground', 'background', 'distant'
    region: Optional[str] = None
    standing_on: Optional[str] = None
    facing: Optional[str] = None
    copied_from: Optional[str] = None
    source: SourceReqsIR = field(default_factory=SourceReqsIR)
    appearance: AppearanceIR = field(default_factory=AppearanceIR)
    transformation: TransformIR = field(default_factory=TransformIR)
    lighting: Optional[LightingIR] = None
    struct_info: Optional[StructIR] = None
    linspace_info: Optional[LinspaceIR] = None
    summon_info: Optional[SummonIR] = None
    is_composite: bool = False
    is_template: bool = False
    linspace_group: Optional[str] = None
    linspace_index: Optional[int] = None
    linspace_total: Optional[int] = None
    summon_group: Optional[str] = None
    summon_index: Optional[int] = None
    summon_total: Optional[int] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ast(cls, node: ObjectNode) -> ObjectIR:
        source = SourceReqsIR.from_ast(node.source)
        appearance = AppearanceIR.from_ast(node.appearance)
        transformation = TransformIR.from_ast(node.transformation)
        lighting = LightingIR.from_ast(node.lighting) if node.lighting else None

        # Prioritize explicit properties over child blocks if set
        facing = node.facing or transformation.facing

        struct_info = None
        if node.struct_call:
            b = node.struct_call.base
            b_str = b if isinstance(b, str) else getattr(b, "name", str(b))
            p_strs = [p if isinstance(p, str) else getattr(p, "name", str(p)) for p in node.struct_call.parts]
            struct_info = StructIR(base=b_str, parts=p_strs, properties=dict(node.struct_call.properties))

        linspace_info = None
        if node.linspace_call:
            t = node.linspace_call.target
            t_str = t if isinstance(t, str) else getattr(t, "name", str(t))
            s_info = None
            if isinstance(t, StructCallNode):
                b_str = t.base if isinstance(t.base, str) else getattr(t.base, "name", str(t.base))
                p_strs = [p if isinstance(p, str) else getattr(p, "name", str(p)) for p in t.parts]
                s_info = StructIR(base=b_str, parts=p_strs, properties=dict(t.properties))
            linspace_info = LinspaceIR(target=t_str, count=node.linspace_call.count, struct_info=s_info, properties=dict(node.linspace_call.properties))

        summon_info = None
        if node.summon_call:
            t = node.summon_call.target
            t_str = t if isinstance(t, str) else getattr(t, "name", str(t))
            s_info = None
            if isinstance(t, StructCallNode):
                b_str = t.base if isinstance(t.base, str) else getattr(t.base, "name", str(t.base))
                p_strs = [p if isinstance(p, str) else getattr(p, "name", str(p)) for p in t.parts]
                s_info = StructIR(base=b_str, parts=p_strs, properties=dict(t.properties))
            summon_info = SummonIR(target=t_str, count=node.summon_call.count, struct_info=s_info, properties=dict(node.summon_call.properties))

        return cls(
            name=node.name,
            depth=node.depth or "midground",
            region=node.region,
            standing_on=node.standing_on,
            facing=facing,
            copied_from=getattr(node, "copied_from", None),
            source=source,
            appearance=appearance,
            transformation=transformation,
            lighting=lighting,
            struct_info=struct_info,
            linspace_info=linspace_info,
            summon_info=summon_info,
            is_composite=bool(struct_info),
            is_template=node.is_template or bool(linspace_info or summon_info),
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
    chain_of_thought: Optional[str] = None

    @classmethod
    def from_ast(cls, node: SceneNode) -> SceneIR:
        camera = CameraIR.from_ast(node.camera)
        environment = EnvironmentIR.from_ast(node.environment)
        objects = {name: ObjectIR.from_ast(obj) for name, obj in node.objects.items()}

        def apply_chained(target_name: str, chained: ChainedCallNode):
            if target_name not in objects:
                return
            obj = objects[target_name]
            for call in chained.calls:
                m = call.method.lower()
                arg = call.args[0] if call.args else None
                if m == "scale":
                    obj.transformation.scale = arg
                elif m == "facing":
                    obj.facing = str(arg)
                    obj.transformation.facing = str(arg)
                elif m in ("rotation", "rotate"):
                    try:
                        obj.transformation.rotation = float(arg)
                    except (ValueError, TypeError):
                        pass
                elif m == "flip":
                    obj.transformation.flip = str(arg)
                elif m == "depth":
                    obj.depth = str(arg)
                elif m == "region":
                    obj.region = str(arg)
                elif m == "standing_on":
                    obj.standing_on = str(arg)
                elif m == "color":
                    obj.appearance.color = str(arg)
                elif m == "brightness":
                    try:
                        obj.appearance.brightness = float(arg)
                    except (ValueError, TypeError):
                        pass
                elif m == "contrast":
                    try:
                        obj.appearance.contrast = float(arg)
                    except (ValueError, TypeError):
                        pass
                elif m == "opacity":
                    try:
                        obj.appearance.opacity = float(arg)
                    except (ValueError, TypeError):
                        pass
                elif m in ("copy", "copy_from", "copied_from"):
                    obj.copied_from = str(arg)
                else:
                    obj.properties[m] = arg

        if hasattr(node, "edits") and node.edits:
            for item in node.edits:
                if isinstance(item, ChainedCallNode):
                    apply_chained(item.target, item)
                elif isinstance(item, ObjectNode):
                    objects[item.name] = ObjectIR.from_ast(item)
                elif isinstance(item, EditBlockNode):
                    for sub in item.items:
                        if isinstance(sub, ChainedCallNode):
                            apply_chained(sub.target, sub)
                        elif isinstance(sub, ObjectNode):
                            objects[sub.name] = ObjectIR.from_ast(sub)
                        elif isinstance(sub, dict):
                            for k, v in sub.items():
                                if isinstance(v, ObjectNode):
                                    objects[k] = ObjectIR.from_ast(v)

        for op_node in getattr(node, "operations", []):
            if hasattr(op_node, "details") and isinstance(op_node.details, ChainedCallNode):
                apply_chained(op_node.details.target, op_node.details)

        # 1. Process scene-level structs from node.structs
        for s_name, s_block in getattr(node, "structs", {}).items():
            if s_name not in objects:
                objects[s_name] = ObjectIR(
                    name=s_name,
                    struct_info=StructIR(base=s_block.base or "", parts=list(s_block.parts), properties=dict(s_block.properties)),
                    is_composite=True,
                    is_template=True,
                )

        # 2. Determine objects explicitly placed in scene relations or constraints
        explicitly_placed = (
            {r.subject for r in node.relations}
            | {r.target for r in node.relations}
            | {c.subject for c in node.constraints}
            | {c.target for c in node.constraints if c.target}
        )

        # 3. Process struct composite objects
        for name, obj in list(objects.items()):
            if obj.struct_info:
                b_name = obj.struct_info.base
                p_names = obj.struct_info.parts
                if b_name:
                    if b_name in objects:
                        if not obj.source.query and objects[b_name].source.query:
                            obj.source.query = objects[b_name].source.query
                        if b_name not in explicitly_placed and not objects[b_name].linspace_group:
                            objects[b_name].is_template = True
                    else:
                        objects[b_name] = ObjectIR(
                            name=b_name,
                            source=SourceReqsIR(query=b_name.replace("_", " ")),
                            is_template=True,
                        )
                for p in p_names:
                    if p in objects:
                        if p not in explicitly_placed and not objects[p].linspace_group:
                            objects[p].is_template = True
                    else:
                        objects[p] = ObjectIR(
                            name=p,
                            source=SourceReqsIR(query=p.replace("_", " ")),
                            is_template=True,
                        )

        # 4. Expand linspace and summon groups into concrete ObjectIR instances
        for name, group_obj in list(objects.items()):
            if group_obj.linspace_info:
                ls = group_obj.linspace_info
                target_name = ls.target
                if ls.struct_info:
                    tmpl_name = f"{group_obj.name}_template"
                    sb = ls.struct_info.base
                    sp = ls.struct_info.parts
                    if sb:
                        if sb in objects:
                            if sb not in explicitly_placed:
                                objects[sb].is_template = True
                        else:
                            objects[sb] = ObjectIR(name=sb, source=SourceReqsIR(query=sb.replace("_", " ")), is_template=True)
                    for p in sp:
                        if p in objects:
                            if p not in explicitly_placed:
                                objects[p].is_template = True
                        else:
                            objects[p] = ObjectIR(name=p, source=SourceReqsIR(query=p.replace("_", " ")), is_template=True)

                    objects[tmpl_name] = ObjectIR(
                        name=tmpl_name,
                        struct_info=ls.struct_info,
                        is_composite=True,
                        is_template=True,
                        source=SourceReqsIR(query=f"{sb} with {sp[0] if sp else 'part'}"),
                        depth=group_obj.depth,
                        region=group_obj.region,
                        standing_on=group_obj.standing_on,
                        transformation=group_obj.transformation,
                        appearance=group_obj.appearance,
                    )
                    target_name = tmpl_name
                elif target_name in objects:
                    if target_name not in explicitly_placed:
                        objects[target_name].is_template = True
                else:
                    objects[target_name] = ObjectIR(
                        name=target_name,
                        source=SourceReqsIR(query=target_name.replace("_", " ")),
                        is_template=True,
                    )

                group_obj.is_template = True
                N = max(1, ls.count)
                for idx in range(N):
                    inst_name = f"{group_obj.name}_{idx + 1}"
                    inst_obj = ObjectIR(
                        name=inst_name,
                        copied_from=target_name,
                        depth=group_obj.depth,
                        region=group_obj.region,
                        standing_on=group_obj.standing_on,
                        facing=group_obj.facing,
                        transformation=TransformIR(
                            scale=group_obj.transformation.scale,
                            facing=group_obj.transformation.facing,
                            rotation=group_obj.transformation.rotation,
                            flip=group_obj.transformation.flip,
                            perspective=group_obj.transformation.perspective,
                            properties=dict(group_obj.transformation.properties),
                        ),
                        appearance=AppearanceIR(
                            color=group_obj.appearance.color,
                            lighting=group_obj.appearance.lighting,
                            brightness=group_obj.appearance.brightness,
                            contrast=group_obj.appearance.contrast,
                            saturation=group_obj.appearance.saturation,
                            opacity=group_obj.appearance.opacity,
                            properties=dict(group_obj.appearance.properties),
                        ),
                        is_template=False,
                        linspace_group=group_obj.name,
                        linspace_index=idx,
                        linspace_total=N,
                    )
                    objects[inst_name] = inst_obj

            elif group_obj.summon_info:
                sm = group_obj.summon_info
                target_name = sm.target
                if sm.struct_info:
                    tmpl_name = f"{group_obj.name}_template"
                    sb = sm.struct_info.base
                    sp = sm.struct_info.parts
                    if sb:
                        if sb in objects:
                            if sb not in explicitly_placed:
                                objects[sb].is_template = True
                        else:
                            objects[sb] = ObjectIR(name=sb, source=SourceReqsIR(query=sb.replace("_", " ")), is_template=True)
                    for p in sp:
                        if p in objects:
                            if p not in explicitly_placed:
                                objects[p].is_template = True
                        else:
                            objects[p] = ObjectIR(name=p, source=SourceReqsIR(query=p.replace("_", " ")), is_template=True)

                    objects[tmpl_name] = ObjectIR(
                        name=tmpl_name,
                        struct_info=sm.struct_info,
                        is_composite=True,
                        is_template=True,
                        source=SourceReqsIR(query=f"{sb} with {sp[0] if sp else 'part'}"),
                        depth=group_obj.depth,
                        region=group_obj.region,
                        standing_on=group_obj.standing_on,
                        transformation=group_obj.transformation,
                        appearance=group_obj.appearance,
                    )
                    target_name = tmpl_name
                elif target_name in objects:
                    if target_name not in explicitly_placed:
                        objects[target_name].is_template = True
                else:
                    objects[target_name] = ObjectIR(
                        name=target_name,
                        source=SourceReqsIR(query=target_name.replace("_", " ")),
                        is_template=True,
                    )

                group_obj.is_template = True
                N = max(1, sm.count)
                for idx in range(N):
                    inst_name = f"{group_obj.name}_{idx + 1}"
                    inst_obj = ObjectIR(
                        name=inst_name,
                        copied_from=target_name,
                        depth=group_obj.depth,
                        region=group_obj.region,
                        standing_on=group_obj.standing_on,
                        facing=group_obj.facing,
                        transformation=TransformIR(
                            scale=group_obj.transformation.scale,
                            facing=group_obj.transformation.facing,
                            rotation=group_obj.transformation.rotation,
                            flip=group_obj.transformation.flip,
                            perspective=group_obj.transformation.perspective,
                            properties=dict(group_obj.transformation.properties),
                        ),
                        appearance=AppearanceIR(
                            color=group_obj.appearance.color,
                            lighting=group_obj.appearance.lighting,
                            brightness=group_obj.appearance.brightness,
                            contrast=group_obj.appearance.contrast,
                            saturation=group_obj.appearance.saturation,
                            opacity=group_obj.appearance.opacity,
                            properties=dict(group_obj.appearance.properties),
                        ),
                        is_template=False,
                        summon_group=group_obj.name,
                        summon_index=idx,
                        summon_total=N,
                    )
                    objects[inst_name] = inst_obj

        # Inherit properties from source object for copied objects
        for name, obj in objects.items():
            if obj.copied_from and obj.copied_from in objects:
                src_obj = objects[obj.copied_from]
                if not obj.source.query and src_obj.source.query:
                    obj.source.query = src_obj.source.query
                if not obj.source.viewpoint and src_obj.source.viewpoint:
                    obj.source.viewpoint = src_obj.source.viewpoint
                if not obj.source.isolated and src_obj.source.isolated:
                    obj.source.isolated = src_obj.source.isolated
                if not obj.source.full_body and src_obj.source.full_body:
                    obj.source.full_body = src_obj.source.full_body
                if not obj.appearance.color and src_obj.appearance.color:
                    obj.appearance.color = src_obj.appearance.color

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
            chain_of_thought=node.chain_of_thought,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_cpp_dsl(self) -> str:
        """Serialize SceneIR back to canonical C++ style Scene DSL."""
        lines: List[str] = []
        if self.chain_of_thought:
            lines.append("// Chain of Thought:")
            for c_line in self.chain_of_thought.strip().splitlines():
                clean = c_line.strip()
                if clean.startswith("//"):
                    clean = clean[2:].strip()
                lines.append(f"// {clean}")
            lines.append("")
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
            if obj.linspace_group or obj.summon_group:
                continue
            if obj.name.endswith("_template") and obj.is_template:
                continue

            if obj.linspace_info:
                ls = obj.linspace_info
                if ls.struct_info:
                    parts_str = ", ".join(ls.struct_info.parts)
                    lines.append(f"        object {name} = linspace(struct({ls.struct_info.base}, {parts_str}), {ls.count}) {{")
                else:
                    lines.append(f"        object {name} = linspace({ls.target}, {ls.count}) {{")
            elif obj.summon_info:
                sm = obj.summon_info
                if sm.struct_info:
                    parts_str = ", ".join(sm.struct_info.parts)
                    lines.append(f"        object {name} = summon(struct({sm.struct_info.base}, {parts_str}), {sm.count}) {{")
                else:
                    lines.append(f"        object {name} = summon({sm.target}, {sm.count}) {{")
            elif obj.struct_info:
                st = obj.struct_info
                parts_str = ", ".join(st.parts)
                lines.append(f"        object {name} = struct({st.base}, {parts_str}) {{")
            elif obj.copied_from:
                lines.append(f"        object {name} = copy({obj.copied_from}) {{")
            else:
                lines.append(f"        object {name} {{")
            # Source
            if not obj.copied_from and any([obj.source.query, obj.source.viewpoint, obj.source.isolated, obj.source.full_body, obj.source.resolution]):
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
