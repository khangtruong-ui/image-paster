"""Parser for C++ style Scene DSL based on Lark."""

from __future__ import annotations
from typing import Any, List, Optional
import lark
from lark import Lark, Tree, Token

from image_paster.dsl.grammar import CPP_SCENE_DSL_GRAMMAR
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


class DSLSyntaxError(Exception):
    """Exception raised when DSL syntax parsing fails."""

    def __init__(self, message: str, line: Optional[int] = None, column: Optional[int] = None, context: Optional[str] = None):
        self.message = message
        self.line = line
        self.column = column
        self.context = context
        pos_info = f" at line {line}, column {column}" if line is not None else ""
        ctx_info = f"\nContext:\n{context}" if context else ""
        super().__init__(f"DSLSyntaxError{pos_info}: {message}{ctx_info}")


def _unwrap_token(val: Any) -> Any:
    if isinstance(val, Token):
        text = str(val)
        if text == "true":
            return True
        if text == "false":
            return False
        # Strip quotes if string literal
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            return text[1:-1]
        # Try int/float
        try:
            if "." in text:
                return float(text)
            return int(text)
        except ValueError:
            return text
    if isinstance(val, Tree):
        if val.data == "true_val":
            return True
        if val.data == "false_val":
            return False
        if val.data == "search_call" and val.children:
            return {"_type": "search_call", "query": _unwrap_token(val.children[0])}
        if val.data == "value" and val.children:
            return _unwrap_token(val.children[0])
    return val


class SceneDSLParser:
    """Pro parser for C++ style Scene DSL."""

    def __init__(self):
        # We use earley parser for maximum grammar flexibility and ambiguity handling
        self._lark = Lark(
            CPP_SCENE_DSL_GRAMMAR,
            start="start",
            parser="earley",
            propagate_positions=True,
        )

    def parse(self, dsl_text: str) -> SceneNode:
        """Parse C++ style DSL text into an AST SceneNode.

        Args:
            dsl_text: The input DSL text in C++ syntax.

        Returns:
            A SceneNode AST root.

        Raises:
            DSLSyntaxError: If syntax or token error occurs.
        """
        try:
            parse_tree = self._lark.parse(dsl_text)
        except lark.exceptions.UnexpectedToken as e:
            context = None
            if hasattr(e, "get_context"):
                try:
                    context = e.get_context(dsl_text)
                except Exception:
                    pass
            raise DSLSyntaxError(
                message=f"Unexpected token '{e.token}'. Expected one of: {e.expected}",
                line=e.line,
                column=e.column,
                context=context,
            ) from e
        except lark.exceptions.UnexpectedCharacters as e:
            context = None
            if hasattr(e, "get_context"):
                try:
                    context = e.get_context(dsl_text)
                except Exception:
                    pass
            raise DSLSyntaxError(
                message=f"Unexpected character at position: {e.char}",
                line=e.line,
                column=e.column,
                context=context,
            ) from e
        except lark.exceptions.LarkError as e:
            raise DSLSyntaxError(message=str(e)) from e

        return self._build_ast(parse_tree)

    def _build_ast(self, tree: Tree) -> SceneNode:
        # tree is start -> scene
        scene_tree = tree.children[0]
        scene_name = "Scene"
        body_trees: List[Tree] = []

        for child in scene_tree.children:
            if isinstance(child, Token):
                scene_name = str(child)
            elif isinstance(child, Tree) and child.data == "scene_body":
                body_trees.append(child.children[0])

        scene_node = SceneNode(name=scene_name)

        for block in body_trees:
            block_type = block.data
            if block_type == "camera_block":
                scene_node.camera = self._parse_camera(block)
            elif block_type == "environment_block":
                scene_node.environment = self._parse_environment(block)
            elif block_type == "objects_block":
                scene_node.objects.update(self._parse_objects(block))
            elif block_type == "relations_block":
                scene_node.relations.extend(self._parse_relations(block))
            elif block_type == "constraints_block":
                scene_node.constraints.extend(self._parse_constraints(block))
            elif block_type == "operations_block":
                scene_node.operations.extend(self._parse_operations(block))

        return scene_node

    def _parse_camera(self, tree: Tree) -> CameraNode:
        props = {}
        for child in tree.children:
            if isinstance(child, Tree) and child.data == "assignment":
                k, v = self._parse_assignment(child)
                props[k] = v
        return CameraNode(properties=props)

    def _parse_lighting(self, tree: Tree) -> LightingNode:
        props = {}
        for child in tree.children:
            if isinstance(child, Tree) and child.data == "assignment":
                k, v = self._parse_assignment(child)
                props[k] = v
        return LightingNode(properties=props)

    def _parse_environment(self, tree: Tree) -> EnvironmentNode:
        props = {}
        lighting = None
        for item in tree.children:
            child = item.children[0]
            if child.data == "assignment":
                k, v = self._parse_assignment(child)
                props[k] = v
            elif child.data == "lighting_block":
                lighting = self._parse_lighting(child)

        env_type = str(props.get("type", "natural"))
        sky = props.get("sky")
        ground = props.get("ground")
        return EnvironmentNode(
            env_type=env_type,
            sky=str(sky) if sky is not None else None,
            ground=str(ground) if ground is not None else None,
            lighting=lighting,
            properties=props,
        )

    def _parse_search_call(self, tree: Tree) -> str:
        if tree.data == "search_call" and tree.children:
            return str(_unwrap_token(tree.children[0]))
        return ""

    def _parse_objects(self, tree: Tree) -> dict[str, ObjectNode]:
        objects = {}
        for obj_tree in tree.children:
            if not isinstance(obj_tree, Tree):
                continue
            rule_name = obj_tree.data
            if rule_name in ("full_object_def", "object_def"):
                obj_name = str(obj_tree.children[0])
                obj_node = ObjectNode(name=obj_name)
                self._populate_object_items(obj_node, obj_tree.children[1:])
                objects[obj_name] = obj_node
            elif rule_name in ("object_search_def", "shorthand_search_def"):
                obj_name = str(obj_tree.children[0])
                search_tree = obj_tree.children[1]
                query = self._parse_search_call(search_tree)
                obj_node = ObjectNode(name=obj_name, source=SourceReqsNode(query=query))
                if len(obj_tree.children) > 2:
                    self._populate_object_items(obj_node, obj_tree.children[2:])
                objects[obj_name] = obj_node
        return objects

    def _populate_object_items(self, obj_node: ObjectNode, items: list) -> None:
        for item in items:
            if not isinstance(item, Tree):
                continue
            child = item.children[0] if item.data == "object_item" else item
            if not isinstance(child, Tree):
                continue
            if child.data == "source_block":
                src_node = self._parse_source_reqs(child)
                if obj_node.source is not None and obj_node.source.query and not src_node.query:
                    src_node.query = obj_node.source.query
                obj_node.source = src_node
            elif child.data == "appearance_block":
                obj_node.appearance = self._parse_appearance(child)
            elif child.data == "transform_block":
                obj_node.transformation = self._parse_transform(child)
            elif child.data == "lighting_block":
                obj_node.lighting = self._parse_lighting(child)
            elif child.data == "search_call_stmt":
                q = self._parse_search_call(child.children[0])
                if obj_node.source is None:
                    obj_node.source = SourceReqsNode(query=q)
                else:
                    obj_node.source.query = q
            elif child.data == "assignment":
                k, v = self._parse_assignment(child)
                if k == "source" and isinstance(v, dict) and v.get("_type") == "search_call":
                    if obj_node.source is None:
                        obj_node.source = SourceReqsNode(query=v["query"])
                    else:
                        obj_node.source.query = v["query"]
                elif k == "query":
                    if obj_node.source is None:
                        obj_node.source = SourceReqsNode(query=str(v))
                    else:
                        obj_node.source.query = str(v)
                else:
                    obj_node.properties[k] = v
                    if k == "depth":
                        obj_node.depth = str(v)
                    elif k == "region":
                        obj_node.region = str(v)
                    elif k == "standing_on":
                        obj_node.standing_on = str(v)
                    elif k == "facing":
                        obj_node.facing = str(v)

    def _parse_source_reqs(self, tree: Tree) -> SourceReqsNode:
        props = {}
        query = None
        for item in tree.children:
            child = item.children[0] if (isinstance(item, Tree) and item.data == "source_item") else item
            if not isinstance(child, Tree):
                continue
            if child.data == "search_call_stmt":
                query = self._parse_search_call(child.children[0])
            elif child.data == "assignment":
                k, v = self._parse_assignment(child)
                if k == "query":
                    query = str(v)
                elif k == "search" and isinstance(v, str):
                    query = v
                elif isinstance(v, dict) and v.get("_type") == "search_call":
                    query = v["query"]
                props[k] = v
        return SourceReqsNode(
            query=query,
            viewpoint=str(props.get("viewpoint")) if "viewpoint" in props else None,
            isolated=str(props.get("isolated")) if "isolated" in props else None,
            full_body=str(props.get("full_body")) if "full_body" in props else None,
            resolution=str(props.get("resolution")) if "resolution" in props else None,
            properties=props,
        )

    def _parse_appearance(self, tree: Tree) -> AppearanceNode:
        props = {}
        for child in tree.children:
            if isinstance(child, Tree) and child.data == "assignment":
                k, v = self._parse_assignment(child)
                props[k] = v
        return AppearanceNode(
            color=str(props.get("color")) if "color" in props else None,
            lighting=str(props.get("lighting")) if "lighting" in props else None,
            brightness=float(props["brightness"]) if "brightness" in props else None,
            contrast=float(props["contrast"]) if "contrast" in props else None,
            saturation=float(props["saturation"]) if "saturation" in props else None,
            opacity=float(props["opacity"]) if "opacity" in props else None,
            properties=props,
        )

    def _parse_transform(self, tree: Tree) -> TransformationNode:
        props = {}
        for child in tree.children:
            if isinstance(child, Tree) and child.data == "assignment":
                k, v = self._parse_assignment(child)
                props[k] = v
        return TransformationNode(
            scale=props.get("scale"),
            facing=str(props.get("facing")) if "facing" in props else None,
            rotation=float(props["rotation"]) if "rotation" in props else None,
            flip=str(props.get("flip")) if "flip" in props else None,
            perspective=str(props.get("perspective")) if "perspective" in props else None,
            properties=props,
        )

    def _parse_assignment(self, tree: Tree) -> tuple[str, Any]:
        key = str(tree.children[0])
        val = _unwrap_token(tree.children[1])
        return key, val

    def _parse_relations(self, tree: Tree) -> List[RelationNode]:
        relations: List[RelationNode] = []
        for item in tree.children:
            child = item.children[0]
            if child.data == "relation_method_call":
                # subject . relation ( target [, params] ) ;
                subj = str(child.children[0])
                rel = str(child.children[1])
                target = ""
                params = []
                if len(child.children) > 2:
                    target = str(_unwrap_token(child.children[2]))
                if len(child.children) > 3:
                    params = [_unwrap_token(c) for c in child.children[3:]]
                relations.append(RelationNode(subject=subj, relation=rel, target=target, params=params))
            elif child.data == "relation_fn_call":
                # rel ( subj , target [, params] ) ;
                rel = str(child.children[0])
                subj = str(_unwrap_token(child.children[1]))
                target = str(_unwrap_token(child.children[2]))
                params = [_unwrap_token(c) for c in child.children[3:]] if len(child.children) > 3 else []
                relations.append(RelationNode(subject=subj, relation=rel, target=target, params=params))
            elif child.data == "relation_block":
                subj = str(child.children[0])
                for assign in child.children[1:]:
                    rel, target = self._parse_assignment(assign)
                    relations.append(RelationNode(subject=subj, relation=rel, target=str(target)))
        return relations

    def _parse_constraints(self, tree: Tree) -> List[ConstraintNode]:
        constraints: List[ConstraintNode] = []
        for item in tree.children:
            child = item.children[0]
            if child.data == "constraint_method_call":
                # subj . constraint ( target [, params] ) ;
                subj = str(child.children[0])
                cstr = str(child.children[1])
                target = None
                params = []
                if len(child.children) > 2:
                    target = str(_unwrap_token(child.children[2]))
                if len(child.children) > 3:
                    params = [_unwrap_token(c) for c in child.children[3:]]
                constraints.append(ConstraintNode(subject=subj, constraint=cstr, target=target, params=params))
            elif child.data == "constraint_fn_call":
                cstr = str(child.children[0])
                subj = str(_unwrap_token(child.children[1]))
                target = str(_unwrap_token(child.children[2])) if len(child.children) > 2 else None
                params = [_unwrap_token(c) for c in child.children[3:]] if len(child.children) > 3 else []
                constraints.append(ConstraintNode(subject=subj, constraint=cstr, target=target, params=params))
            elif child.data == "constraint_block":
                subj = str(child.children[0])
                for assign in child.children[1:]:
                    cstr, target = self._parse_assignment(assign)
                    constraints.append(ConstraintNode(subject=subj, constraint=cstr, target=str(target)))
        return constraints

    def _parse_operations(self, tree: Tree) -> List[OperationNode]:
        ops: List[OperationNode] = []
        for item in tree.children:
            name = str(item.children[0])
            ops.append(OperationNode(name=name))
        return ops
