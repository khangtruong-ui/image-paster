"""Semantic validator for parsed Scene DSL AST."""

from __future__ import annotations
from typing import List, Set, Dict
from collections import defaultdict

from image_paster.dsl.ast_nodes import SceneNode, RelationNode, ConstraintNode


class DSLValidationError(Exception):
    """Exception raised when DSL semantic validation fails."""

    def __init__(self, message: str, errors: List[str] | None = None):
        self.message = message
        self.errors = errors or [message]
        formatted = "\n  - " + "\n  - ".join(self.errors)
        super().__init__(f"DSLValidationError: {message}{formatted}")


KNOWN_ENVIRONMENT_ANCHORS = {
    "ground",
    "floor",
    "sky",
    "camera",
    "canvas",
    "background",
    "foreground",
    "midground",
    "center",
    "table",
    "wall",
    "viewer",
}

VALID_DEPTHS = {"distant", "background", "midground", "foreground"}
VALID_FACING = {"left", "right", "toward_camera", "away", "front", "side", "center"}
VALID_REGIONS = {
    "left", "right", "center", "top", "bottom",
    "top_left", "top_right", "bottom_left", "bottom_right",
    "foreground", "background", "midground",
}


class DSLValidator:
    """Validates the semantic consistency of a SceneNode AST."""

    def __init__(self, allow_empty_objects: bool = False):
        self.allow_empty_objects = allow_empty_objects

    def validate(self, scene: SceneNode) -> None:
        """Validate scene AST semantics.

        Args:
            scene: The parsed SceneNode.

        Raises:
            DSLValidationError: If any semantic violation is detected.
        """
        errors: List[str] = []

        # 1. Objects check
        if not scene.objects and not self.allow_empty_objects:
            errors.append("Scene contains no objects defined in 'objects { ... }'. At least one object is required.")

        defined_objects: Set[str] = set(scene.objects.keys())

        # Check each object properties
        for name, obj in scene.objects.items():
            if obj.depth and obj.depth.lower() not in VALID_DEPTHS:
                errors.append(
                    f"Object '{name}' has invalid depth '{obj.depth}'. Valid options: {sorted(VALID_DEPTHS)}"
                )
            if obj.facing and obj.facing.lower() not in VALID_FACING:
                errors.append(
                    f"Object '{name}' has invalid facing '{obj.facing}'. Valid options: {sorted(VALID_FACING)}"
                )
            if obj.region and obj.region.lower() not in VALID_REGIONS:
                errors.append(
                    f"Object '{name}' has invalid region '{obj.region}'. Valid options: {sorted(VALID_REGIONS)}"
                )

        # 2. Check relations
        rel_pairs: Dict[tuple[str, str], Set[str]] = defaultdict(set)
        depth_graph: Dict[str, Set[str]] = defaultdict(set)  # X -> Y means X is behind Y (Y is closer than X)

        for rel in scene.relations:
            # Subject must be a defined object
            if rel.subject not in defined_objects:
                errors.append(
                    f"Relation subject '{rel.subject}' in '{rel.subject}.{rel.relation}({rel.target})' is not a defined object."
                )

            # Target must be either a defined object or an environment anchor
            if rel.target and rel.target not in defined_objects and rel.target not in KNOWN_ENVIRONMENT_ANCHORS:
                errors.append(
                    f"Relation target '{rel.target}' in '{rel.subject}.{rel.relation}({rel.target})' is neither a defined object nor a known environment anchor."
                )

            pair = (rel.subject, rel.target)
            rel_type = rel.relation.lower()
            rel_pairs[pair].add(rel_type)

            # Check direct spatial contradictions for same pair
            if "left_of" in rel_pairs[pair] and "right_of" in rel_pairs[pair]:
                errors.append(f"Contradictory relations: '{rel.subject}' is both left_of and right_of '{rel.target}'.")
            if "above" in rel_pairs[pair] and "below" in rel_pairs[pair]:
                errors.append(f"Contradictory relations: '{rel.subject}' is both above and below '{rel.target}'.")
            if "behind" in rel_pairs[pair] and "in_front_of" in rel_pairs[pair]:
                errors.append(f"Contradictory relations: '{rel.subject}' is both behind and in_front_of '{rel.target}'.")

            # Check symmetric contradictions (A left_of B and B left_of A)
            reverse_pair = (rel.target, rel.subject)
            if rel_type == "left_of" and "left_of" in rel_pairs[reverse_pair]:
                errors.append(f"Contradictory symmetric relations: '{rel.subject}' left_of '{rel.target}' and vice-versa.")
            if rel_type == "above" and "above" in rel_pairs[reverse_pair]:
                errors.append(f"Contradictory symmetric relations: '{rel.subject}' above '{rel.target}' and vice-versa.")
            if rel_type == "behind" and "behind" in rel_pairs[reverse_pair]:
                errors.append(f"Contradictory symmetric relations: '{rel.subject}' behind '{rel.target}' and vice-versa.")

            # Record depth ordering
            if rel.target in defined_objects:
                if rel_type in ("behind", "further_than"):
                    # subject is behind target -> subject < target in depth
                    depth_graph[rel.subject].add(rel.target)
                elif rel_type in ("in_front_of", "closer_than"):
                    # target is behind subject
                    depth_graph[rel.target].add(rel.subject)

        # 3. Check depth cycle detection (e.g. A behind B, B behind C, C behind A)
        cycle = self._detect_cycle(depth_graph)
        if cycle:
            cycle_str = " -> ".join(cycle)
            errors.append(f"Cyclic depth relationship detected: {cycle_str}")

        # 4. Check constraints
        for cstr in scene.constraints:
            if cstr.subject not in defined_objects:
                errors.append(
                    f"Constraint subject '{cstr.subject}' in '{cstr.subject}.{cstr.constraint}(...)' is not a defined object."
                )
            if cstr.target and cstr.target not in defined_objects and cstr.target not in KNOWN_ENVIRONMENT_ANCHORS:
                errors.append(
                    f"Constraint target '{cstr.target}' in '{cstr.subject}.{cstr.constraint}({cstr.target})' is neither a defined object nor an environment anchor."
                )

        if errors:
            raise DSLValidationError(f"Found {len(errors)} semantic validation errors in Scene DSL", errors)

    def _detect_cycle(self, graph: Dict[str, Set[str]]) -> List[str] | None:
        """Detect cycle in directed graph using DFS. Returns cycle path if found."""
        visited: Dict[str, int] = {}  # 0: unvisited, 1: visiting, 2: visited
        parent: Dict[str, str | None] = {}

        nodes = list(graph.keys())
        for node in nodes:
            visited[node] = 0

        path: List[str] = []

        def dfs(u: str) -> bool:
            visited[u] = 1
            path.append(u)
            for v in graph.get(u, set()):
                if visited.get(v, 0) == 1:
                    # Found cycle
                    cycle_start = path.index(v)
                    path.append(v)
                    return True
                if visited.get(v, 0) == 0:
                    if dfs(v):
                        return True
            path.pop()
            visited[u] = 2
            return False

        for node in nodes:
            if visited[node] == 0:
                if dfs(node):
                    return path

        return None
