"""LLM Scene Planner for generating C++ style Scene DSL from natural language prompts."""

from __future__ import annotations
import gc
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, List, Dict, Any

from image_paster.dsl import parse_dsl, SceneIR, DSLSyntaxError, DSLValidationError
from image_paster.llm.prompts.system_prompt import SYSTEM_PROMPT
from image_paster.llm.prompts.few_shot_examples import FEW_SHOT_EXAMPLES

logger = logging.getLogger(__name__)

# Default model ladder in the 2-5B parameter range for <12GB VRAM GPUs
DEFAULT_2_TO_5B_MODELS = [
    "google/gemma-4-E2B",   # Primary 2-5B model
    "Qwen/Qwen3.5-2B",      # Compact fallback
]


class PlannerError(Exception):
    """Exception raised when scene planning fails."""
    pass


def extract_dsl_from_response(text: str) -> str:
    """Extract C++ style DSL from LLM output (which might be wrapped in ```cpp ... ```).

    Robust against trailing prompt continuations or hallucinated second scenes
    from completion models (e.g. google/gemma-4-E2B).
    """
    text = text.strip()

    # 1. Check for markdown code block containing scene definition
    code_block_match = re.search(r"```(?:cpp|c\+\+|dsl)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    search_target = text
    if code_block_match and "scene " in code_block_match.group(1):
        search_target = code_block_match.group(1).strip()

    # 2. Find the scene definition start
    scene_match = re.search(r"\bscene\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\{", search_target)
    if not scene_match:
        # Fallback to general regex search if no explicit scene keyword
        raw_match = re.search(r"(scene\s+[a-zA-Z_][^{]*\{.*\})", search_target, re.DOTALL)
        return raw_match.group(1).strip() if raw_match else text

    scene_open_brace = search_target.find("{", scene_match.start())
    if scene_open_brace == -1:
        return text

    preceding = search_target[:scene_match.start()]
    # Capture comments or preceding struct definitions before 'scene'
    comments_match = re.search(r"((?:(?:\s*//[^\n]*\n|\s*/\*.*?\*/\s*)+))\s*$", preceding, re.DOTALL)
    if comments_match and comments_match.group(1).strip():
        start_offset = comments_match.start(1)
    else:
        struct_match = re.search(r"(\bstruct\s+[a-zA-Z_].*)$", preceding, re.DOTALL)
        if struct_match:
            start_offset = struct_match.start(1)
        else:
            start_offset = scene_match.start()

    # 3. Find matching closing brace for scene using a robust scanner
    depth = 0
    in_string = False
    escape = False
    in_line_comment = False
    in_block_comment = False
    end_idx = -1

    i = scene_open_brace
    while i < len(search_target):
        char = search_target[i]
        if escape:
            escape = False
            i += 1
            continue
        if char == "\\" and in_string:
            escape = True
            i += 1
            continue
        if in_string:
            if char == "\"":
                in_string = False
            i += 1
            continue
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if char == "*" and i + 1 < len(search_target) and search_target[i + 1] == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if char == "\"":
            in_string = True
            i += 1
            continue
        if char == "/" and i + 1 < len(search_target):
            if search_target[i + 1] == "/":
                in_line_comment = True
                i += 2
                continue
            elif search_target[i + 1] == "*":
                in_block_comment = True
                i += 2
                continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
        i += 1

    if end_idx != -1:
        return search_target[start_offset:end_idx].strip()
    return search_target[start_offset:].strip()


class BaseScenePlanner(ABC):
    """Abstract base for scene planners."""

    debug: bool = False
    last_raw_response: Optional[str] = None
    fallback_occurred: bool = False
    fallback_reason: Optional[str] = None
    primary_model_name: Optional[str] = None

    @abstractmethod
    def plan(self, prompt: str) -> Tuple[str, SceneIR]:
        """Generate C++ style DSL and SceneIR from prompt.

        Args:
            prompt: User natural language prompt.

        Returns:
            Tuple of (dsl_text, scene_ir).
        """
        pass

    def replan(
        self,
        prompt: str,
        previous_dsl: str,
        failure_reasons: List[str],
    ) -> Tuple[str, SceneIR]:
        """Re-plan scene DSL when retrieval or segmentation fails."""
        return self.plan(prompt)

    def adjust_dsl(
        self,
        existing_dsl: str,
        adjustment_prompt: str,
    ) -> Tuple[str, SceneIR]:
        """Adjust an existing Scene DSL based on user instructions."""
        raise NotImplementedError("adjust_dsl not implemented for this planner")


class RuleBasedPlanner(BaseScenePlanner):
    """Deterministic, offline semantic planner.

    Parses common natural language scenes, identifies primary entities, spatial relations,
    environment, and constraints, and produces valid C++ Scene DSL with rich, elaborate search queries.
    """

    KNOWN_ENVIRONMENTS = [
        "forest", "woods", "desert", "beach", "ocean", "mountain", "snow",
        "city", "street", "room", "kitchen", "park", "garden", "spaceship",
        "sky", "studio"
    ]

    RELATION_KEYWORDS = {
        "behind": "behind",
        "in front of": "in_front_of",
        "next to": "near",
        "near": "near",
        "beside": "near",
        "left of": "left_of",
        "to the left of": "left_of",
        "right of": "right_of",
        "to the right of": "right_of",
        "above": "above",
        "below": "below",
        "under": "below",
        "underneath": "below",
        "on top of": "standing_on",
        "sitting on": "standing_on",
        "standing on": "standing_on",
        "inside": "inside",
    }

    def __init__(self, creative: bool = True, debug: bool = False):
        self.creative = creative
        self.debug = debug
        self.last_raw_response: Optional[str] = None
        self.fallback_occurred: bool = False
        self.fallback_reason: Optional[str] = None
        self.primary_model_name: Optional[str] = "rule_based"

    def plan(self, prompt: str) -> Tuple[str, SceneIR]:
        cleaned = prompt.strip().lower()

        # 1. Detect environment
        env_type = "natural"
        ground_type = "ground"
        for env in self.KNOWN_ENVIRONMENTS:
            if env in cleaned:
                env_type = env
                if env in ("forest", "woods", "park", "garden"):
                    ground_type = "grassy"
                elif env in ("beach", "desert"):
                    ground_type = "sand"
                elif env in ("city", "street"):
                    ground_type = "pavement"
                elif env in ("room", "kitchen"):
                    ground_type = "floor"
                elif env == "spaceship":
                    ground_type = "metal_deck"
                break

        # Natural, concise search queries for background retrieval
        env_queries = {
            "forest": "dense misty pine forest landscape",
            "woods": "misty autumn woods landscape",
            "desert": "desert sand dunes under open sky",
            "beach": "tropical beach with turquoise ocean",
            "ocean": "deep blue open ocean with horizon",
            "mountain": "snow-capped alpine mountain peak",
            "snow": "winter snowy landscape with pine trees",
            "city": "modern city street",
            "street": "urban street sidewalk",
            "room": "cozy living room interior",
            "kitchen": "modern kitchen interior",
            "park": "sunny green park landscape",
            "garden": "blooming flower garden",
            "spaceship": "futuristic spaceship command bridge interior",
            "sky": "clear blue sky with soft white clouds",
            "studio": "clean studio backdrop",
        }
        env_query = env_queries.get(env_type, f"{env_type} landscape")

        # 2. Extract objects & spatial relation
        detected_relation = None
        rel_key_found = None
        for key, rel in self.RELATION_KEYWORDS.items():
            if f" {key} " in f" {cleaned} ":
                detected_relation = rel
                rel_key_found = key
                break

        # Simple semantic extraction
        words = re.findall(r"\b[a-zA-Z_]+\b", cleaned)
        stop_words = {"a", "an", "the", "in", "on", "at", "inside", "standing", "sitting", "lying", "of", "and", "with", "is"}

        obj1 = "subject"
        obj2 = None

        if rel_key_found:
            parts = cleaned.split(rel_key_found, 1)
            left_words = [w for w in re.findall(r"\b[a-zA-Z_]+\b", parts[0]) if w not in stop_words and w not in self.KNOWN_ENVIRONMENTS]
            right_words = [w for w in re.findall(r"\b[a-zA-Z_]+\b", parts[1]) if w not in stop_words and w not in self.KNOWN_ENVIRONMENTS]
            if left_words:
                obj1 = "_".join(left_words[:2])
            if right_words:
                obj2 = "_".join(right_words[:2])
        else:
            cand = [w for w in words if w not in stop_words and w not in self.KNOWN_ENVIRONMENTS]
            if "car" in cand:
                car_idx = cand.index("car")
                obj1 = "_".join(cand[:car_idx + 1])
                if len(cand) > car_idx + 1:
                    obj2 = cand[car_idx + 1]
            elif cand:
                obj1 = cand[0]
                if len(cand) > 1:
                    obj2 = cand[1]

        # Clean identifiers
        obj1 = re.sub(r"\W+", "_", obj1).strip("_") or "subject"
        if obj2:
            obj2 = re.sub(r"\W+", "_", obj2).strip("_")

        # Determine depths and layout
        obj1_depth = "midground" if (detected_relation == "behind" and obj2) else "foreground"
        obj2_depth = "foreground" if (detected_relation == "behind" and obj2) else "background"
        obj1_region = 'right' if detected_relation in ('behind', 'right_of') else 'center'

        # Contextual decorative creative objects (if creative mode enabled)
        creative_items = []
        if self.creative:
            creative_presets = {
                "forest": [
                    ("wildflowers", "wildflowers on grass", "foreground", "bottom_right", "small"),
                    ("mossy_rock", "mossy forest boulder", "foreground", "bottom_left", "small"),
                ],
                "woods": [
                    ("bush", "green forest bush", "foreground", "bottom_left", "medium"),
                    ("wildflowers", "woodland wildflowers", "foreground", "bottom_right", "small"),
                ],
                "park": [
                    ("wildflowers", "park flowers", "foreground", "bottom_right", "small"),
                    ("park_bench", "wooden park bench", "midground", "bottom_left", "medium"),
                ],
                "garden": [
                    ("potted_plant", "potted plant", "foreground", "bottom_right", "medium"),
                    ("garden_lantern", "decorative garden lantern", "foreground", "bottom_left", "small"),
                ],
                "beach": [
                    ("seashells", "sea shells on beach sand", "foreground", "bottom_right", "small"),
                    ("beach_umbrella", "colorful beach umbrella", "midground", "bottom_left", "medium"),
                ],
                "desert": [
                    ("small_cactus", "small desert cactus", "background", "bottom_left", "small"),
                    ("sandstone_rock", "desert sandstone boulder", "foreground", "bottom_right", "medium"),
                ],
                "mountain": [
                    ("pine_tree", "pine tree", "foreground", "bottom_left", "medium"),
                    ("alpine_flowers", "alpine mountain flowers", "foreground", "bottom_right", "small"),
                ],
                "snow": [
                    ("snowy_rock", "rock covered with snow", "foreground", "bottom_left", "small"),
                    ("snowy_evergreen", "snow covered evergreen pine", "midground", "bottom_right", "medium"),
                ],
                "city": [
                    ("street_lamp", "street lamp post", "background", "bottom_left", "medium"),
                    ("bench", "city park bench", "foreground", "bottom_right", "medium"),
                ],
                "street": [
                    ("fire_hydrant", "red fire hydrant on sidewalk", "foreground", "bottom_left", "small"),
                    ("street_sign", "metal street sign post", "midground", "bottom_right", "medium"),
                ],
                "room": [
                    ("houseplant", "potted houseplant", "background", "bottom_right", "medium"),
                    ("floor_lamp", "standing floor lamp", "midground", "bottom_left", "medium"),
                ],
                "kitchen": [
                    ("fruit_bowl", "bowl filled with fresh fruit", "background", "bottom_left", "small"),
                    ("coffee_mug", "ceramic coffee mug", "foreground", "bottom_right", "tiny"),
                ],
                "spaceship": [
                    ("terminal_panel", "sci-fi computer terminal console", "background", "bottom_left", "medium"),
                    ("cargo_crate", "futuristic sci-fi cargo container", "foreground", "bottom_right", "medium"),
                ],
            }
            presets = creative_presets.get(
                env_type,
                [("wildflowers", "wildflowers on grass", "foreground", "bottom_right", "small")]
            )
            for c_name, c_query, c_depth, c_region, c_scale in presets:
                if c_name not in (obj1, obj2):
                    creative_items.append((c_name, c_query, c_depth, c_region, c_scale))

        # Check scene tone (dark, night, dim vs daylight)
        is_dark = any(w in cleaned for w in ("dark", "night", "dusk", "evening", "dim", "moonlight", "shadow"))
        lighting_temp = "cool" if is_dark else "warm"
        lighting_intensity = "soft" if is_dark else "medium"

        # Build natural, concise object search query without bloating
        clean_name = obj1.replace('_', ' ')
        if "car" in clean_name and "road" in cleaned:
            obj1_query = f"{clean_name} on the road"
        else:
            obj1_query = f"{clean_name} full body"

        # Build Chain of Thought reasoning comments demonstrating logical deductions & creative scene population
        cot_lines = [
            "// Chain of Thought:",
            f"// Visualizing scene for '{prompt}' in a {env_type} setting with focal subject '{obj1}'.",
        ]
        if is_dark:
            cot_lines.append("// It is a dark scene so I should make the trees dim, setting cool night temperature and reduced brightness to match the night ambiance.")
        else:
            cot_lines.append(f"// Natural daylight environment ({env_type}) with {lighting_temp} illumination.")

        if creative_items:
            c_name, c_query, c_depth, c_region, c_scale = creative_items[0]
            if env_type in ("mountain", "forest", "woods") and "tree" in c_name:
                cot_lines.append(f"// Contextual Logic: I believe the scene of a {env_type} should have trees, so I place {c_name} in the {c_region}.")
            else:
                cot_lines.append(f"// Contextual Logic: I believe the scene of a {env_type} should have {c_name.replace('_', ' ')}, so I add {c_name} to the {c_region}.")
            if len(creative_items) > 1:
                extra_names = [it[0].replace('_', ' ') for it in creative_items[1:]]
                cot_lines.append(f"// Populating the scene with additional contextual elements ({', '.join(extra_names)}) to fill the image.")
        else:
            cot_lines.append("// Composition Logic: Focusing directly on prompt-specified entities with grounded anchoring.")

        if obj2:
            cot_lines.append(f"// Object Relations: Position {obj1} in relation to {obj2} ({detected_relation}).")

        dsl_lines = [
            f"// Generated Scene DSL for: {prompt}",
            *cot_lines,
            "",
            f"scene GeneratedScene {{",
            f"    camera {{",
            f"        viewpoint = eye_level;",
            f"        perspective = natural;",
            f"        focus = {obj1};",
            f"    }}",
            f"",
            f"    environment {{",
            f'        search("{env_query}");',
            f'        type = "{env_type}";',
            f'        ground = "{ground_type}";',
            f"        lighting {{",
            f"            direction = upper_left;",
            f"            intensity = {lighting_intensity};",
            f"            temperature = {lighting_temp};",
            f"        }}",
            f"    }}",
            f"",
            f"    objects {{",
            f"        object {obj1} {{",
            f"            source {{",
            f'                search("{obj1_query}");',
            f"                viewpoint = side;",
            f"                full_body = required;",
            f"                isolated = preferred;",
            f"            }}",
            f"            depth = {obj1_depth};",
            f"            region = {obj1_region};",
            f"            standing_on = {obj2 if (detected_relation == 'standing_on' and obj2) else 'ground'};",
            f"            appearance {{",
            f"                lighting = inherit_scene;",
            f"                brightness = {-0.2 if is_dark else 0.0};",
            f"            }}",
            f"            transformation {{",
            f"                scale = {'medium' if (detected_relation == 'standing_on' and obj2) else 'large'};",
            f"                facing = right;",
            f"            }}",
            f"        }}",
            f"    }}",
        ]

        # Fix objects closing brace when obj2 or creative_items exist
        dsl_lines.pop()  # remove premature closing brace

        if obj2 and obj2 != obj1:
            obj2_query = f"{obj2.replace('_', ' ')} full body"
            dsl_lines.extend([
                f"        object {obj2} {{",
                f"            source {{",
                f'                search("{obj2_query}");',
                f"                viewpoint = frontal;",
                f"                isolated = preferred;",
                f"            }}",
                f"            depth = {obj2_depth};",
                f"            region = left;",
                f"            standing_on = ground;",
                f"            appearance {{",
                f"                brightness = {-0.2 if is_dark else 0.0};",
                f"            }}",
                f"            transformation {{",
                f"                scale = large;",
                f"            }}",
                f"        }}",
            ])

        for c_name, c_query, c_depth, c_region, c_scale in creative_items:
            dsl_lines.extend([
                f"        object {c_name} {{",
                f"            source {{",
                f'                search("{c_query}");',
                f"                viewpoint = frontal;",
                f"                isolated = preferred;",
                f"            }}",
                f"            depth = {c_depth};",
                f"            region = {c_region};",
                f"            standing_on = ground;",
                f"            appearance {{",
                f"                brightness = {-0.25 if is_dark else 0.0};",
                f"            }}",
                f"            transformation {{",
                f"                scale = {c_scale};",
                f"            }}",
                f"        }}",
            ])

        dsl_lines.append("    }")

        dsl_lines.extend([
            "",
            "    relations {",
        ])

        if obj2 and obj2 != obj1 and detected_relation:
            dsl_lines.append(f"        {obj1}.{detected_relation}({obj2});")
        if detected_relation != "standing_on":
            dsl_lines.append(f"        {obj1}.standing_on(ground);")
        if obj2 and obj2 != obj1:
            dsl_lines.append(f"        {obj2}.standing_on(ground);")
        for c_name, _, _, _, _ in creative_items:
            dsl_lines.append(f"        {c_name}.standing_on(ground);")

        dsl_lines.extend([
            f"    }}",
            f"",
            f"    constraints {{",
        ])
        if detected_relation == "standing_on" and obj2:
            dsl_lines.append(f"        {obj1}.must_touch({obj2});")
        else:
            dsl_lines.append(f"        {obj1}.must_touch(ground);")
        if obj2 and obj2 != obj1:
            dsl_lines.append(f"        {obj2}.must_touch(ground);")
            if detected_relation == "behind":
                dsl_lines.append(f"        {obj2}.must_occlude({obj1});")
        for c_name, _, _, _, _ in creative_items:
            dsl_lines.append(f"        {c_name}.must_touch(ground);")

        dsl_lines.extend([
            f"    }}",
            f"",
            f"    operations {{",
            f"        retrieve;",
            f"        segment;",
            f"        solve_layout;",
            f"        compose;",
            f"        blend;",
            f"        verify;",
            f"    }}",
            f"}}",
        ])

        dsl_text = "\n".join(dsl_lines)
        self.last_raw_response = dsl_text
        scene_ir = parse_dsl(dsl_text, validate=True)
        return dsl_text, scene_ir

    def replan(
        self,
        prompt: str,
        previous_dsl: str,
        failure_reasons: List[str],
    ) -> Tuple[str, SceneIR]:
        """Adjust previous DSL to address retrieval/segmentation failures."""
        try:
            ir = parse_dsl(previous_dsl, validate=False)
        except Exception:
            return self.plan(prompt)

        for reason in failure_reasons:
            for obj_name, obj in ir.objects.items():
                if f"'{obj_name}'" in reason or obj_name in reason:
                    # Simplify the query dramatically
                    clean = obj_name.replace("_", " ")
                    obj.source.query = clean
                    obj.source.isolated = "preferred"
                    obj.source.full_body = "preferred"

        new_dsl = ir.to_cpp_dsl()
        try:
            validated_ir = parse_dsl(new_dsl, validate=True)
            return new_dsl, validated_ir
        except Exception:
            return new_dsl, ir

    def adjust_dsl(
        self,
        existing_dsl: str,
        adjustment_prompt: str,
    ) -> Tuple[str, SceneIR]:
        """Adjust compiled DSL according to natural language modification instructions."""
        ir = parse_dsl(existing_dsl, validate=False)
        prompt_lower = adjustment_prompt.strip().lower()

        # Identify target object if specified
        target_obj = None
        for name in ir.objects.keys():
            if name.lower() in prompt_lower or name.replace("_", " ") in prompt_lower:
                target_obj = name
                break
        if not target_obj and ir.objects:
            target_obj = next(iter(ir.objects.keys()))

        if target_obj and target_obj in ir.objects:
            obj = ir.objects[target_obj]

            # Position / height adjustments
            if any(w in prompt_lower for w in ("higher", "put it higher", "move up", "too low")):
                if obj.region in ("bottom", "bottom_left", "bottom_right"):
                    obj.region = "center"
                else:
                    obj.region = "top"
                obj.standing_on = None

            elif any(w in prompt_lower for w in ("lower", "put it lower", "move down", "too high")):
                obj.region = "bottom"
                obj.standing_on = "ground"

            elif any(w in prompt_lower for w in ("to the left", "move left", "on the left", "left")):
                obj.region = "left"

            elif any(w in prompt_lower for w in ("to the right", "move right", "on the right", "right")):
                obj.region = "right"

            elif any(w in prompt_lower for w in ("in the center", "move to center", "middle")):
                obj.region = "center"

            # Scale / size adjustments
            if any(w in prompt_lower for w in ("bigger", "larger", "increase size", "too small")):
                scale_map = {"tiny": "small", "small": "medium", "medium": "large", "large": "huge"}
                if isinstance(obj.transformation.scale, str):
                    obj.transformation.scale = scale_map.get(obj.transformation.scale, "large")
                elif isinstance(obj.transformation.scale, (int, float)):
                    obj.transformation.scale = float(obj.transformation.scale) * 1.3
            elif any(w in prompt_lower for w in ("smaller", "decrease size", "too big", "too large")):
                scale_map = {"huge": "large", "large": "medium", "medium": "small", "small": "tiny"}
                if isinstance(obj.transformation.scale, str):
                    obj.transformation.scale = scale_map.get(obj.transformation.scale, "small")
                elif isinstance(obj.transformation.scale, (int, float)):
                    obj.transformation.scale = float(obj.transformation.scale) * 0.7

            # Facing adjustments
            if "face left" in prompt_lower or "facing left" in prompt_lower or "turn left" in prompt_lower:
                obj.facing = "left"
                obj.transformation.facing = "left"
            elif "face right" in prompt_lower or "facing right" in prompt_lower or "turn right" in prompt_lower:
                obj.facing = "right"
                obj.transformation.facing = "right"

            # Depth adjustments
            if any(w in prompt_lower for w in ("closer", "bring forward", "foreground")):
                obj.depth = "foreground"
            elif any(w in prompt_lower for w in ("further", "farther", "behind", "background")):
                obj.depth = "background"

            # Copy / duplicate instruction
            if any(w in prompt_lower for w in ("copy", "duplicate", "add another")):
                new_name = f"{target_obj}_copy"
                from image_paster.dsl.ir import ObjectIR, TransformIR
                new_obj = ObjectIR(
                    name=new_name,
                    depth="midground" if obj.depth == "foreground" else "foreground",
                    region="right" if obj.region != "right" else "left",
                    standing_on=obj.standing_on,
                    facing=obj.facing,
                    copied_from=target_obj,
                    source=obj.source,
                    appearance=obj.appearance,
                    transformation=TransformIR(scale=obj.transformation.scale, facing=obj.transformation.facing),
                )
                ir.objects[new_name] = new_obj
                from image_paster.dsl.ir import RelationIR, ConstraintIR
                ir.relations.append(RelationIR(subject=new_name, relation="right_of", target=target_obj))
                ir.constraints.append(ConstraintIR(subject=new_name, constraint="must_touch", target="ground"))

            # Remove object instruction
            if any(w in prompt_lower for w in ("remove", "delete", "get rid of")):
                del ir.objects[target_obj]
                ir.relations = [r for r in ir.relations if r.subject != target_obj and r.target != target_obj]
                ir.constraints = [c for c in ir.constraints if c.subject != target_obj and c.target != target_obj]

        new_dsl = ir.to_cpp_dsl()
        try:
            validated_ir = parse_dsl(new_dsl, validate=True)
            return new_dsl, validated_ir
        except Exception:
            return new_dsl, ir


class TransformersPlanner(BaseScenePlanner):
    """Local Hugging Face transformers scene planner using open-weights LLMs.

    Employs a multi-tier fallback ladder in the 2-5B parameter range for <12GB VRAM GPUs:
        1. Primary: 'google/gemma-4-E2B'
        2. Fallback on OOM / error: 'Qwen/Qwen3.5-2B'
        3. Deterministic offline fallback: RuleBasedPlanner
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_candidates: Optional[List[str]] = None,
        device: Optional[str] = None,
        torch_dtype: Any = None,
        creative: bool = True,
        fallback_planner: Optional[BaseScenePlanner] = None,
        max_retries: int = 2,
        hf_token: Optional[str] = None,
        debug: bool = False,
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        self.creative = creative
        self.debug = debug
        self.fallback_planner = fallback_planner or RuleBasedPlanner(creative=creative, debug=debug)
        self.max_retries = max_retries
        self.hf_token = hf_token
        self._pipeline = None
        self.active_model_name: Optional[str] = None
        self.fallback_occurred: bool = False
        self.fallback_reason: Optional[str] = None
        self.last_raw_response: Optional[str] = None

        # Build candidate model ladder
        if model_candidates:
            self.model_candidates = list(model_candidates)
        elif model_name:
            user_models = [m.strip() for m in model_name.split(",") if m.strip()]
            self.model_candidates = user_models + [m for m in DEFAULT_2_TO_5B_MODELS if m not in user_models]
        else:
            self.model_candidates = list(DEFAULT_2_TO_5B_MODELS)

        self.primary_model_name = self.model_candidates[0] if self.model_candidates else "google/gemma-4-E2B"

    def _load_model(self, model_name: str):
        """Load a specific model and return a text-generation pipeline."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype = self.torch_dtype
        if dtype is None:
            dtype = torch.float16 if (torch.cuda.is_available() and device != "cpu") else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.hf_token,
            trust_remote_code=True,
            clean_up_tokenization_spaces=False,
        )

        # Check if accelerate is available for device_map="auto"
        has_accelerate = False
        try:
            import accelerate
            has_accelerate = True
        except ImportError:
            pass

        device_map = "auto" if (has_accelerate and device != "cpu") else None
        model = None

        if device_map is not None:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    token=self.hf_token,
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.debug(f"Loading '{model_name}' with device_map failed ({e}). Retrying direct device placement...")
                model = None

        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=None,
                low_cpu_mem_usage=False,
                token=self.hf_token,
                trust_remote_code=True,
            )
            if device != "cpu":
                model = model.to(device)

        if device == "cpu":
            model = model.to("cpu")

        # Disable conflicting max_length in generation_config if present
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.max_length = None

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        if hasattr(pipe, "generation_config") and pipe.generation_config is not None:
            pipe.generation_config.max_length = None
        if hasattr(tokenizer, "clean_up_tokenization_spaces"):
            tokenizer.clean_up_tokenization_spaces = False
        return pipe

    def _generate_text(self, pipe, system_msg: str, user_prompt: str, max_new_tokens: int = 700) -> str:
        """Robust text generation supporting models with or without chat templates."""
        tok = getattr(pipe, "tokenizer", None)
        has_chat_template = getattr(tok, "chat_template", None) is not None

        if has_chat_template:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ]
            try:
                output = pipe(messages, max_new_tokens=max_new_tokens, do_sample=False, clean_up_tokenization_spaces=False)
                res = output[0]["generated_text"]
                content = res[-1].get("content", str(res[-1])) if isinstance(res, list) else str(res)
                self.last_raw_response = content
                return content
            except Exception as e:
                logger.debug(f"Chat template generation failed ({e}); falling back to text prompt format.")

        full_prompt = f"{system_msg}\n\n{user_prompt}\n\nC++ Scene DSL:\n"
        output = pipe(full_prompt, max_new_tokens=max_new_tokens, do_sample=False, clean_up_tokenization_spaces=False)
        generated = output[0]["generated_text"]
        if isinstance(generated, str):
            if generated.startswith(full_prompt):
                generated = generated[len(full_prompt):].strip()
            else:
                generated = generated.strip()
        elif isinstance(generated, list):
            generated = generated[-1].get("content", str(generated))
        else:
            generated = str(generated)

        # Truncate any hallucinated prompt sequence continuation
        for stop_seq in ("\nUser Prompt:", "\nPrompt:", "\nHere are examples of C++ Scene DSL:"):
            idx = generated.find(stop_seq)
            if idx != -1:
                generated = generated[:idx].strip()

        self.last_raw_response = generated
        return generated

    def _get_pipeline(self):
        """Retrieve or load a pipeline, iterating down the candidate ladder on failure."""
        if self._pipeline is not None:
            return self._pipeline

        import torch

        while self.model_candidates:
            cand = self.model_candidates[0]
            try:
                logger.info(f"Attempting to load transformers model '{cand}'...")
                self._pipeline = self._load_model(cand)
                self.active_model_name = cand
                logger.info(f"Successfully loaded '{cand}' into memory.")
                return self._pipeline
            except (torch.cuda.OutOfMemoryError, Exception) as e:
                err_msg = f"Failed to load transformers model '{cand}': {e}"
                logger.warning(f"{err_msg}. Clearing VRAM and falling back down model ladder...")
                if cand == self.primary_model_name:
                    self.fallback_occurred = True
                    self.fallback_reason = err_msg
                    if self.debug:
                        raise RuntimeError(f"Model fallback occurred in debug mode: {err_msg}") from e

                self.model_candidates.pop(0)
                self._pipeline = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        self.fallback_occurred = True
        self.fallback_reason = self.fallback_reason or "All transformer model candidates exhausted."
        if self.debug:
            raise RuntimeError(f"Model fallback occurred in debug mode: {self.fallback_reason}")

        logger.warning("All transformer model candidates exhausted. Falling back to RuleBasedPlanner.")
        return None

    def plan(self, prompt: str) -> Tuple[str, SceneIR]:
        import torch

        while True:
            pipe = self._get_pipeline()
            if pipe is None:
                self.fallback_occurred = True
                self.fallback_reason = self.fallback_reason or "Pipeline is None"
                if self.debug:
                    raise RuntimeError(f"Model fallback occurred in debug mode: {self.fallback_reason}")
                return self.fallback_planner.plan(prompt)

            if self.active_model_name != self.primary_model_name:
                self.fallback_occurred = True
                self.fallback_reason = (
                    f"Fell back from primary model '{self.primary_model_name}' "
                    f"to candidate '{self.active_model_name}'"
                )
                if self.debug:
                    raise RuntimeError(f"Model fallback occurred in debug mode: {self.fallback_reason}")

            creative_clause = (
                "Creative Mode is ACTIVE (default): Feel free to be expressive and creative! Dense scene population is encouraged: "
                "add rich contextual objects, props, foreground accents, and basic geometric shapes or text labels where fitting "
                "to create a vibrant, complete, and well-filled composition."
                if self.creative
                else "Prompt-Only Mode is ACTIVE: Generate ONLY the objects explicitly mentioned in the prompt. Do NOT add extra decorative objects."
            )

            system_msg = f"{SYSTEM_PROMPT}\n\n{creative_clause}"
            few_shot_str = "\n\n".join(
                f"User Prompt: {ex['prompt']}\nC++ Scene DSL:\n{ex['dsl']}"
                for ex in FEW_SHOT_EXAMPLES
            )
            base_user_prompt = (
                f"Here are examples of C++ Scene DSL:\n\n{few_shot_str}\n\n"
                f"Now generate valid C++ Scene DSL for:\nPrompt: {prompt}"
            )

            current_user_prompt = base_user_prompt
            oom_encountered = False

            for attempt in range(self.max_retries + 1):
                try:
                    resp_text = self._generate_text(pipe, system_msg, current_user_prompt, max_new_tokens=700)
                    self.last_raw_response = resp_text
                    dsl_text = extract_dsl_from_response(resp_text)
                    scene_ir = parse_dsl(dsl_text, validate=True)
                    return dsl_text, scene_ir
                except torch.cuda.OutOfMemoryError as e:
                    logger.warning(
                        f"CUDA OutOfMemoryError during generation with '{self.active_model_name}': {e}. "
                        "Freeing VRAM and falling back to smaller model..."
                    )
                    oom_encountered = True
                    self.fallback_occurred = True
                    self.fallback_reason = f"CUDA OutOfMemoryError during generation with '{self.active_model_name}': {e}"
                    if self.debug:
                        raise RuntimeError(f"Model fallback occurred in debug mode: {self.fallback_reason}") from e
                    break
                except (DSLSyntaxError, DSLValidationError) as e:
                    logger.debug(f"TransformersPlanner attempt {attempt + 1} validation error: {e}")
                    current_user_prompt = (
                        f"{base_user_prompt}\n\n"
                        f"Your previous attempt produced a validation error:\n{str(e)}\n"
                        f"Please correct the error and output valid C++ Scene DSL only."
                    )
                except Exception as e:
                    logger.warning(f"TransformersPlanner generation failed with error: {e}")
                    break

            if oom_encountered:
                # Evict current model, clear memory, and try next model
                if self.model_candidates and self.model_candidates[0] == self.active_model_name:
                    self.model_candidates.pop(0)
                self._pipeline = None
                self.active_model_name = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue

            # Fall back to deterministic planner if retries exhausted without OOM
            self.fallback_occurred = True
            self.fallback_reason = f"Generation retries exhausted with '{self.active_model_name}'"
            if self.debug:
                raise RuntimeError(f"Model fallback occurred in debug mode: {self.fallback_reason}")
            return self.fallback_planner.plan(prompt)

    def replan(
        self,
        prompt: str,
        previous_dsl: str,
        failure_reasons: List[str],
    ) -> Tuple[str, SceneIR]:
        """Re-plan scene DSL when retrieval or segmentation fails."""
        pipe = self._get_pipeline()
        if pipe is None:
            self.fallback_occurred = True
            self.fallback_reason = self.fallback_reason or "Pipeline is None for replan"
            if self.debug:
                raise RuntimeError(f"Model fallback occurred in debug mode: {self.fallback_reason}")
            return self.fallback_planner.replan(prompt, previous_dsl, failure_reasons)

        failure_text = "\n".join(f"- {r}" for r in failure_reasons)
        replan_user_prompt = (
            f"Initial prompt: {prompt}\n\n"
            f"Previous C++ Scene DSL:\n```cpp\n{previous_dsl}\n```\n\n"
            f"Retrieval / segmentation failed with the following issues:\n{failure_text}\n\n"
            f"Please adjust the scene DSL to fix retrieval. Simplify the search queries (e.g. 'a red car on the road'). "
            f"Output ONLY the complete updated C++ Scene DSL."
        )
        try:
            resp_text = self._generate_text(pipe, SYSTEM_PROMPT, replan_user_prompt, max_new_tokens=700)
            self.last_raw_response = resp_text
            dsl_text = extract_dsl_from_response(resp_text)
            scene_ir = parse_dsl(dsl_text, validate=True)
            return dsl_text, scene_ir
        except Exception as e:
            err_msg = f"TransformersPlanner.replan failed: {e}"
            logger.warning(f"{err_msg}. Falling back to RuleBasedPlanner.")
            self.fallback_occurred = True
            self.fallback_reason = err_msg
            if self.debug:
                raise RuntimeError(f"Model fallback occurred in debug mode: {err_msg}") from e
            return self.fallback_planner.replan(prompt, previous_dsl, failure_reasons)

    def adjust_dsl(
        self,
        existing_dsl: str,
        adjustment_prompt: str,
    ) -> Tuple[str, SceneIR]:
        """Adjust an existing Scene DSL based on user feedback/prompt."""
        pipe = self._get_pipeline()
        if pipe is None:
            self.fallback_occurred = True
            self.fallback_reason = self.fallback_reason or "Pipeline is None for adjust_dsl"
            if self.debug:
                raise RuntimeError(f"Model fallback occurred in debug mode: {self.fallback_reason}")
            return self.fallback_planner.adjust_dsl(existing_dsl, adjustment_prompt)

        user_msg = (
            f"Here is an existing C++ Scene DSL:\n```cpp\n{existing_dsl}\n```\n\n"
            f"User adjustment request:\n\"{adjustment_prompt}\"\n\n"
            f"Please modify the C++ Scene DSL according to the request. Output ONLY the complete updated C++ Scene DSL."
        )
        try:
            resp_text = self._generate_text(pipe, SYSTEM_PROMPT, user_msg, max_new_tokens=700)
            self.last_raw_response = resp_text
            dsl_text = extract_dsl_from_response(resp_text)
            scene_ir = parse_dsl(dsl_text, validate=True)
            return dsl_text, scene_ir
        except Exception as e:
            err_msg = f"TransformersPlanner.adjust_dsl failed: {e}"
            logger.warning(f"{err_msg}. Falling back to RuleBasedPlanner.")
            self.fallback_occurred = True
            self.fallback_reason = err_msg
            if self.debug:
                raise RuntimeError(f"Model fallback occurred in debug mode: {err_msg}") from e
            return self.fallback_planner.adjust_dsl(existing_dsl, adjustment_prompt)


class LLMScenePlanner(BaseScenePlanner):
    """LLM-powered scene planner using external LLM or custom provider callable."""

    def __init__(
        self,
        llm_fn: Optional[Callable[[str, str], str]] = None,
        fallback_planner: Optional[BaseScenePlanner] = None,
        creative: bool = True,
        max_retries: int = 2,
        debug: bool = False,
    ):
        """Initialize LLM scene planner.

        Args:
            llm_fn: Callable(system_prompt, user_prompt) -> str.
            fallback_planner: Fallback planner if llm_fn fails or is None.
            creative: Whether to add contextual decorative elements.
            max_retries: Number of retries on syntax/validation error.
            debug: Whether debug mode is active (raising errors on fallback).
        """
        self.llm_fn = llm_fn
        self.debug = debug
        self.fallback_planner = fallback_planner or RuleBasedPlanner(creative=creative, debug=debug)
        self.creative = creative
        self.max_retries = max_retries
        self.fallback_occurred: bool = False
        self.fallback_reason: Optional[str] = None
        self.last_raw_response: Optional[str] = None
        self.primary_model_name: Optional[str] = "external_llm"

    def plan(self, prompt: str) -> Tuple[str, SceneIR]:
        if not self.llm_fn:
            self.fallback_occurred = True
            self.fallback_reason = "No llm_fn provided"
            if self.debug:
                raise RuntimeError(f"Model fallback occurred in debug mode: {self.fallback_reason}")
            return self.fallback_planner.plan(prompt)

        creative_clause = (
            "Creative Mode is ACTIVE (default): Feel free to be expressive and creative! Dense scene population is encouraged: "
            "add rich contextual objects, props, foreground accents, and basic geometric shapes or text labels where fitting "
            "to create a vibrant, complete, and well-filled composition."
            if self.creative
            else "Prompt-Only Mode is ACTIVE: Generate ONLY the objects explicitly mentioned in the prompt. Do NOT add extra decorative objects."
        )

        system_msg = f"{SYSTEM_PROMPT}\n\n{creative_clause}"
        few_shot_str = "\n\n".join(
            f"User Prompt: {ex['prompt']}\nC++ Scene DSL:\n{ex['dsl']}"
            for ex in FEW_SHOT_EXAMPLES
        )
        full_user_prompt = f"Here are examples:\n\n{few_shot_str}\n\nNow generate C++ Scene DSL for:\nPrompt: {prompt}"

        last_error = None
        current_user_prompt = full_user_prompt

        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm_fn(system_msg, current_user_prompt)
                self.last_raw_response = response
                dsl_text = extract_dsl_from_response(response)
                scene_ir = parse_dsl(dsl_text, validate=True)
                return dsl_text, scene_ir
            except (DSLSyntaxError, DSLValidationError) as e:
                last_error = e
                # Retry with error feedback
                current_user_prompt = (
                    f"{full_user_prompt}\n\n"
                    f"Your previous attempt produced an error:\n{str(e)}\n"
                    f"Please fix the error and output valid C++ Scene DSL only."
                )
            except Exception as e:
                last_error = e
                break

        # Fallback if LLM repeatedly fails
        self.fallback_occurred = True
        self.fallback_reason = f"LLMScenePlanner failed after retries: {last_error}"
        if self.debug:
            raise RuntimeError(f"Model fallback occurred in debug mode: {self.fallback_reason}")
        if self.fallback_planner:
            return self.fallback_planner.plan(prompt)

        raise PlannerError(f"Failed to plan scene for '{prompt}': {last_error}")

    def replan(
        self,
        prompt: str,
        previous_dsl: str,
        failure_reasons: List[str],
    ) -> Tuple[str, SceneIR]:
        """Re-plan scene DSL when retrieval or segmentation fails."""
        if not self.llm_fn:
            self.fallback_occurred = True
            self.fallback_reason = "No llm_fn provided for replan"
            if self.debug:
                raise RuntimeError(f"Model fallback occurred in debug mode: {self.fallback_reason}")
            return self.fallback_planner.replan(prompt, previous_dsl, failure_reasons)

        failure_text = "\n".join(f"- {r}" for r in failure_reasons)
        user_msg = (
            f"Initial prompt: {prompt}\n\n"
            f"Previous C++ Scene DSL:\n```cpp\n{previous_dsl}\n```\n\n"
            f"Retrieval / segmentation failed with the following issues:\n{failure_text}\n\n"
            f"Please adjust the search queries or object definitions in the C++ Scene DSL to fix retrieval. "
            f"Keep search queries simple and natural (e.g. 'a red car on the road'). "
            f"Output ONLY the complete updated C++ Scene DSL."
        )
        try:
            resp = self.llm_fn(SYSTEM_PROMPT, user_msg)
            self.last_raw_response = resp
            dsl_text = extract_dsl_from_response(resp)
            scene_ir = parse_dsl(dsl_text, validate=True)
            return dsl_text, scene_ir
        except Exception as e:
            err_msg = f"LLMScenePlanner.replan failed: {e}"
            logger.warning(f"{err_msg}. Falling back.")
            self.fallback_occurred = True
            self.fallback_reason = err_msg
            if self.debug:
                raise RuntimeError(f"Model fallback occurred in debug mode: {err_msg}") from e
            return self.fallback_planner.replan(prompt, previous_dsl, failure_reasons)

    def adjust_dsl(
        self,
        existing_dsl: str,
        adjustment_prompt: str,
    ) -> Tuple[str, SceneIR]:
        """Adjust an existing Scene DSL based on user feedback/prompt."""
        if not self.llm_fn:
            self.fallback_occurred = True
            self.fallback_reason = "No llm_fn provided for adjust_dsl"
            if self.debug:
                raise RuntimeError(f"Model fallback occurred in debug mode: {self.fallback_reason}")
            return self.fallback_planner.adjust_dsl(existing_dsl, adjustment_prompt)

        user_msg = (
            f"Here is an existing C++ Scene DSL:\n```cpp\n{existing_dsl}\n```\n\n"
            f"User adjustment request:\n\"{adjustment_prompt}\"\n\n"
            f"Please modify the C++ Scene DSL according to the request. Output ONLY the complete updated C++ Scene DSL."
        )
        try:
            resp = self.llm_fn(SYSTEM_PROMPT, user_msg)
            self.last_raw_response = resp
            dsl_text = extract_dsl_from_response(resp)
            scene_ir = parse_dsl(dsl_text, validate=True)
            return dsl_text, scene_ir
        except Exception as e:
            err_msg = f"LLMScenePlanner.adjust_dsl failed: {e}"
            logger.warning(f"{err_msg}. Falling back.")
            self.fallback_occurred = True
            self.fallback_reason = err_msg
            if self.debug:
                raise RuntimeError(f"Model fallback occurred in debug mode: {err_msg}") from e
            return self.fallback_planner.adjust_dsl(existing_dsl, adjustment_prompt)


def create_llm_planner(
    provider: str = "transformers",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    creative: bool = True,
    max_retries: int = 2,
    hf_token: Optional[str] = None,
    debug: bool = False,
) -> BaseScenePlanner:
    """Factory helper to instantiate an LLM scene planner with common providers.

    Supported providers:
        - "transformers": Local Hugging Face pipeline with automatic multi-tier OOM fallback
                          (default: "google/gemma-4-E2B" -> "Qwen/Qwen3.5-2B").
        - "rule_based" / "offline": Built-in deterministic semantic planner (no GPU or API keys required).
        - "openai": OpenAI ChatCompletion (e.g. model="gpt-4o", model="gpt-4o-mini").
        - "gemini": Google Gemini API (e.g. model="gemini-1.5-flash").
        - "auto": Defaults to local transformers model ladder with graceful fallback.
    """
    provider_lower = provider.lower()

    if provider_lower in ("rule_based", "offline"):
        return RuleBasedPlanner(creative=creative, debug=debug)

    if provider_lower in ("transformers", "auto"):
        try:
            return TransformersPlanner(
                model_name=model or "google/gemma-4-E2B",
                creative=creative,
                fallback_planner=RuleBasedPlanner(creative=creative, debug=debug),
                max_retries=max_retries,
                hf_token=hf_token,
                debug=debug,
            )
        except Exception as e:
            if debug:
                raise RuntimeError(f"Model fallback occurred in debug mode: failed to initialize TransformersPlanner: {e}") from e
            logger.warning(f"Could not initialize TransformersPlanner: {e}. Falling back to RuleBasedPlanner.")
            p = RuleBasedPlanner(creative=creative, debug=debug)
            p.fallback_occurred = True
            p.fallback_reason = f"Failed to initialize TransformersPlanner: {e}"
            return p

    if provider_lower == "openai":
        try:
            import openai
            client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
            m = model or "gpt-4o"

            def openai_fn(sys_prompt: str, user_prompt: str) -> str:
                res = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )
                return res.choices[0].message.content or ""

            return LLMScenePlanner(
                llm_fn=openai_fn,
                fallback_planner=RuleBasedPlanner(creative=creative, debug=debug),
                creative=creative,
                max_retries=max_retries,
                debug=debug,
            )
        except ImportError as e:
            if debug:
                raise RuntimeError(f"Model fallback occurred in debug mode: openai module not found: {e}") from e
            p = RuleBasedPlanner(creative=creative, debug=debug)
            p.fallback_occurred = True
            p.fallback_reason = "openai module not installed"
            return p

    if provider_lower == "gemini":
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
            m = model or "gemini-1.5-flash"
            g_model = genai.GenerativeModel(m)

            def gemini_fn(sys_prompt: str, user_prompt: str) -> str:
                full_prompt = f"{sys_prompt}\n\n---\n\n{user_prompt}"
                resp = g_model.generate_content(full_prompt)
                return resp.text or ""

            return LLMScenePlanner(
                llm_fn=gemini_fn,
                fallback_planner=RuleBasedPlanner(creative=creative, debug=debug),
                creative=creative,
                max_retries=max_retries,
                debug=debug,
            )
        except ImportError as e:
            if debug:
                raise RuntimeError(f"Model fallback occurred in debug mode: google.generativeai module not found: {e}") from e
            p = RuleBasedPlanner(creative=creative, debug=debug)
            p.fallback_occurred = True
            p.fallback_reason = "google.generativeai module not installed"
            return p

    return RuleBasedPlanner(creative=creative, debug=debug)
