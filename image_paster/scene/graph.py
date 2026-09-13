"""Scene Graph data structures for representing entities, spatial relations, and constraints."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any

from image_paster.dsl.ir import SceneIR, ObjectIR, RelationIR, ConstraintIR


@dataclass
class GraphNode:
    """A node in the scene graph (an object or environment anchor)."""
    name: str
    is_anchor: bool = False
    object_ir: Optional[ObjectIR] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """A directed edge representing a relation or constraint."""
    source: str
    target: str
    edge_type: str  # 'relation' or 'constraint'
    name: str       # e.g., 'behind', 'left_of', 'must_touch'
    params: List[Any] = field(default_factory=list)


class SceneGraph:
    """Directed scene graph capturing semantic relationships and constraints."""

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._adj: Dict[str, List[GraphEdge]] = {}

    def add_node(self, name: str, is_anchor: bool = False, object_ir: Optional[ObjectIR] = None) -> GraphNode:
        node = GraphNode(name=name, is_anchor=is_anchor, object_ir=object_ir)
        self.nodes[name] = node
        if name not in self._adj:
            self._adj[name] = []
        return node

    def add_edge(self, source: str, target: str, edge_type: str, name: str, params: Optional[List[Any]] = None) -> GraphEdge:
        if source not in self.nodes:
            self.add_node(source, is_anchor=False)
        if target not in self.nodes:
            self.add_node(target, is_anchor=True)

        edge = GraphEdge(source=source, target=target, edge_type=edge_type, name=name, params=params or [])
        self.edges.append(edge)
        self._adj[source].append(edge)
        return edge

    def get_outgoing_edges(self, node_name: str) -> List[GraphEdge]:
        return self._adj.get(node_name, [])

    def get_incoming_edges(self, node_name: str) -> List[GraphEdge]:
        return [e for e in self.edges if e.target == node_name]

    @classmethod
    def from_scene_ir(cls, scene: SceneIR) -> SceneGraph:
        """Construct SceneGraph from SceneIR."""
        sg = cls()

        # Add environment anchors
        anchors = ["ground", "sky", "camera", "canvas", "background", "foreground"]
        for anchor in anchors:
            sg.add_node(anchor, is_anchor=True)

        # Add objects
        for name, obj_ir in scene.objects.items():
            sg.add_node(name, is_anchor=False, object_ir=obj_ir)

        # Add relations
        for rel in scene.relations:
            sg.add_edge(
                source=rel.subject,
                target=rel.target,
                edge_type="relation",
                name=rel.relation,
                params=rel.params,
            )

        # Add constraints
        for cstr in scene.constraints:
            if cstr.target:
                sg.add_edge(
                    source=cstr.subject,
                    target=cstr.target,
                    edge_type="constraint",
                    name=cstr.constraint,
                    params=cstr.params,
                )

        return sg
