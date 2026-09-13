"""LLM Scene Planner for generating C++ style Scene DSL from natural language prompts."""

from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, List, Dict, Any

from image_paster.dsl import parse_dsl, SceneIR, DSLSyntaxError, DSLValidationError
from image_paster.llm.prompts.system_prompt import SYSTEM_PROMPT
from image_paster.llm.prompts.few_shot_examples import FEW_SHOT_EXAMPLES


class PlannerError(Exception):
    """Exception raised when scene planning fails."""
    pass


def extract_dsl_from_response(text: str) -> str:
    """Extract C++ style DSL from LLM output (which might be wrapped in ```cpp ... ```)."""
    text = text.strip()
    # Check for markdown code blocks
    code_block_match = re.search(r"```(?:cpp|c\+\+|dsl)?\s*(scene\s+[^{]+\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Check for raw scene block
    scene_match = re.search(r"(scene\s+[^{]+\{.*\})", text, re.DOTALL)
    if scene_match:
        return scene_match.group(1).strip()

    return text


class BaseScenePlanner(ABC):
    """Abstract base for scene planners."""

    @abstractmethod
    def plan(self, prompt: str) -> Tuple[str, SceneIR]:
        """Generate C++ style DSL and SceneIR from prompt.

        Args:
            prompt: User natural language prompt.

        Returns:
            Tuple of (dsl_text, scene_ir).
        """
        pass


class RuleBasedPlanner(BaseScenePlanner):
    """Deterministic, offline semantic planner.

    Parses common natural language scenes, identifies primary entities, spatial relations,
    environment, and constraints, and produces valid C++ Scene DSL.
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

        # 2. Extract objects & spatial relation
        # Example: "an elephant standing behind a tree in a forest"
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
            if cand:
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

        dsl_lines = [
            f"// Generated Scene DSL for: {prompt}",
            f"scene GeneratedScene {{",
            f"    camera {{",
            f"        viewpoint = eye_level;",
            f"        perspective = natural;",
            f"        focus = {obj1};",
            f"    }}",
            f"",
            f"    environment {{",
            f'        type = "{env_type}";',
            f'        ground = "{ground_type}";',
            f"        lighting {{",
            f"            direction = upper_left;",
            f"            intensity = medium;",
            f"            temperature = warm;",
            f"        }}",
            f"    }}",
            f"",
            f"    objects {{",
            f"        object {obj1} {{",
            f"            source {{",
            f"                viewpoint = side;",
            f"                full_body = required;",
            f"                isolated = preferred;",
            f"            }}",
            f"            depth = {obj1_depth};",
            f"            region = {'right' if detected_relation in ('behind', 'right_of') else 'center'};",
            f"            standing_on = {obj2 if (detected_relation == 'standing_on' and obj2) else 'ground'};",
            f"            appearance {{",
            f"                lighting = inherit_scene;",
            f"            }}",
            f"            transformation {{",
            f"                scale = {'medium' if (detected_relation == 'standing_on' and obj2) else 'large'};",
            f"                facing = right;",
            f"            }}",
            f"        }}",
        ]

        if obj2 and obj2 != obj1:
            dsl_lines.extend([
                f"        object {obj2} {{",
                f"            source {{",
                f"                viewpoint = frontal;",
                f"                isolated = preferred;",
                f"            }}",
                f"            depth = {obj2_depth};",
                f"            region = left;",
                f"            standing_on = ground;",
                f"            transformation {{",
                f"                scale = large;",
                f"            }}",
                f"        }}",
            ])

        dsl_lines.extend([
            f"    }}",
            f"",
            f"    relations {{",
        ])

        if obj2 and obj2 != obj1 and detected_relation:
            dsl_lines.append(f"        {obj1}.{detected_relation}({obj2});")
        if detected_relation != "standing_on":
            dsl_lines.append(f"        {obj1}.standing_on(ground);")
        if obj2 and obj2 != obj1:
            dsl_lines.append(f"        {obj2}.standing_on(ground);")

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
        scene_ir = parse_dsl(dsl_text, validate=True)
        return dsl_text, scene_ir


class LLMScenePlanner(BaseScenePlanner):
    """LLM-powered scene planner using external LLM or custom provider callable."""

    def __init__(
        self,
        llm_fn: Optional[Callable[[str, str], str]] = None,
        fallback_planner: Optional[BaseScenePlanner] = None,
        max_retries: int = 2,
    ):
        """Initialize LLM scene planner.

        Args:
            llm_fn: Callable(system_prompt, user_prompt) -> str.
            fallback_planner: Fallback planner if llm_fn fails or is None.
            max_retries: Number of retries on syntax/validation error.
        """
        self.llm_fn = llm_fn
        self.fallback_planner = fallback_planner or RuleBasedPlanner()
        self.max_retries = max_retries

    def plan(self, prompt: str) -> Tuple[str, SceneIR]:
        if not self.llm_fn:
            return self.fallback_planner.plan(prompt)

        # Build full prompt
        system_msg = SYSTEM_PROMPT
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
        if self.fallback_planner:
            return self.fallback_planner.plan(prompt)

        raise PlannerError(f"Failed to plan scene for '{prompt}': {last_error}")
