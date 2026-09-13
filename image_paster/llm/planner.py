"""LLM Scene Planner for generating C++ style Scene DSL from natural language prompts."""

from __future__ import annotations
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, List, Dict, Any

from image_paster.dsl import parse_dsl, SceneIR, DSLSyntaxError, DSLValidationError
from image_paster.llm.prompts.system_prompt import SYSTEM_PROMPT
from image_paster.llm.prompts.few_shot_examples import FEW_SHOT_EXAMPLES

logger = logging.getLogger(__name__)


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

    def __init__(self, creative: bool = True):
        self.creative = creative

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

        env_queries = {
            "forest": "lush green pine forest landscape photo",
            "woods": "misty deep woods autumn landscape photo",
            "desert": "vast desert sand dunes landscape photo",
            "beach": "tropical sunny ocean beach landscape photo",
            "ocean": "deep blue open ocean water landscape photo",
            "mountain": "majestic snow-capped mountain landscape photo",
            "snow": "winter snowy landscape with pine trees photo",
            "city": "bustling modern city street architecture photo",
            "street": "urban street sidewalk architecture photo",
            "room": "modern cozy living room interior photo",
            "kitchen": "bright modern kitchen interior photo",
            "park": "sunny green public park landscape photo",
            "garden": "vibrant blooming botanical garden landscape photo",
            "spaceship": "sci-fi futuristic spaceship cabin interior",
            "sky": "clear blue sky with soft white clouds photo",
            "studio": "clean studio backdrop background",
        }
        env_query = env_queries.get(env_type, f"{env_type} landscape background photo")

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
        obj1_region = 'right' if detected_relation in ('behind', 'right_of') else 'center'

        # Contextual decorative creative objects (if creative mode enabled)
        creative_item = None
        if self.creative:
            creative_presets = {
                "forest": ("wildflowers", "small cluster of forest wildflowers isolated", "foreground", "bottom_right"),
                "woods": ("bush", "small green forest shrub bush isolated", "foreground", "bottom_left"),
                "park": ("wildflowers", "small colorful park flowers isolated", "foreground", "bottom_right"),
                "garden": ("potted_plant", "small flowering potted plant isolated", "foreground", "bottom_right"),
                "beach": ("seashells", "collection of sea shells on sand isolated", "foreground", "bottom_right"),
                "desert": ("small_cactus", "small desert cactus plant isolated", "background", "bottom_left"),
                "mountain": ("pine_sapling", "small baby pine tree sapling isolated", "background", "bottom_left"),
                "snow": ("snowy_rock", "small rock covered with snow isolated", "foreground", "bottom_left"),
                "city": ("street_lamp", "vintage street lamp post isolated", "background", "bottom_left"),
                "street": ("fire_hydrant", "red fire hydrant on pavement isolated", "foreground", "bottom_left"),
                "room": ("houseplant", "small green indoor houseplant potted isolated", "background", "bottom_right"),
                "kitchen": ("fruit_bowl", "small decorative ceramic fruit bowl isolated", "background", "bottom_left"),
                "spaceship": ("terminal_panel", "small sci-fi computer terminal console isolated", "background", "bottom_left"),
            }
            c_name, c_query, c_depth, c_region = creative_presets.get(
                env_type, ("wildflowers", "small cluster of wildflowers isolated", "foreground", "bottom_right")
            )
            if c_name in (obj1, obj2):
                c_name, c_query, c_depth, c_region = ("pebbles", "small cluster of stones and pebbles isolated", "foreground", "bottom_left")
            if obj1_region == "right" or (obj2 and obj1_region == "center"):
                c_region = "bottom_left"
            else:
                c_region = "bottom_right"
            creative_item = (c_name, c_query, c_depth, c_region)

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
            f'        search("{env_query}");',
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
            f'                search("{obj1.replace("_", " ")}");',
            f"                viewpoint = side;",
            f"                full_body = required;",
            f"                isolated = preferred;",
            f"            }}",
            f"            depth = {obj1_depth};",
            f"            region = {obj1_region};",
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
                f'                search("{obj2.replace("_", " ")}");',
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

        if creative_item:
            c_name, c_query, c_depth, c_region = creative_item
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
                f"            transformation {{",
                f"                scale = small;",
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
        if creative_item:
            dsl_lines.append(f"        {creative_item[0]}.standing_on(ground);")

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
        if creative_item:
            dsl_lines.append(f"        {creative_item[0]}.must_touch(ground);")

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


class TransformersPlanner(BaseScenePlanner):
    """Local Hugging Face transformers scene planner using an open-weights LLM.

    Defaults to 'Qwen/Qwen2.5-1.5B-Instruct', consuming ~2.9GB VRAM in fp16,
    comfortably operating on <12GB VRAM GPUs alongside SAM 3 (<1.7GB VRAM).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        torch_dtype: Any = None,
        creative: bool = True,
        fallback_planner: Optional[BaseScenePlanner] = None,
        max_retries: int = 2,
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.creative = creative
        self.fallback_planner = fallback_planner or RuleBasedPlanner(creative=creative)
        self.max_retries = max_retries
        self._pipeline = None
        self._load_failed = False

    def _get_pipeline(self):
        if self._pipeline is None and not self._load_failed:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

                device = self.device
                if device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                dtype = self.torch_dtype
                if dtype is None:
                    dtype = torch.float16 if (torch.cuda.is_available() and device != "cpu") else torch.float32

                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map="auto" if device != "cpu" else None,
                    low_cpu_mem_usage=True,
                )
                if device == "cpu":
                    model = model.to("cpu")

                self._pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                )
            except Exception as e:
                logger.warning(
                    f"Could not load transformers model '{self.model_name}': {e}. "
                    "Falling back to RuleBasedPlanner."
                )
                self._load_failed = True
                self._pipeline = None
        return self._pipeline

    def plan(self, prompt: str) -> Tuple[str, SceneIR]:
        pipe = self._get_pipeline()
        if pipe is None:
            return self.fallback_planner.plan(prompt)

        creative_clause = (
            "Creative Mode is ACTIVE (default): In addition to the primary subjects, add 1-2 small contextual "
            "decorative objects on the background/ground (e.g. wildflowers, bush, small rocks) with scale = small."
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

        for attempt in range(self.max_retries + 1):
            try:
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": current_user_prompt},
                ]
                output = pipe(messages, max_new_tokens=512, do_sample=False)
                resp_text = output[0]["generated_text"][-1]["content"]
                dsl_text = extract_dsl_from_response(resp_text)
                scene_ir = parse_dsl(dsl_text, validate=True)
                return dsl_text, scene_ir
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

        return self.fallback_planner.plan(prompt)


class LLMScenePlanner(BaseScenePlanner):
    """LLM-powered scene planner using external LLM or custom provider callable."""

    def __init__(
        self,
        llm_fn: Optional[Callable[[str, str], str]] = None,
        fallback_planner: Optional[BaseScenePlanner] = None,
        creative: bool = True,
        max_retries: int = 2,
    ):
        """Initialize LLM scene planner.

        Args:
            llm_fn: Callable(system_prompt, user_prompt) -> str.
            fallback_planner: Fallback planner if llm_fn fails or is None.
            creative: Whether to add contextual decorative elements.
            max_retries: Number of retries on syntax/validation error.
        """
        self.llm_fn = llm_fn
        self.fallback_planner = fallback_planner or RuleBasedPlanner(creative=creative)
        self.creative = creative
        self.max_retries = max_retries

    def plan(self, prompt: str) -> Tuple[str, SceneIR]:
        if not self.llm_fn:
            return self.fallback_planner.plan(prompt)

        creative_clause = (
            "Creative Mode is ACTIVE (default): In addition to the primary subjects, add 1-2 small contextual "
            "decorative objects on the background/ground (e.g. wildflowers, bush, small rocks) with scale = small."
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


def create_llm_planner(
    provider: str = "transformers",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    creative: bool = True,
    max_retries: int = 2,
) -> BaseScenePlanner:
    """Factory helper to instantiate an LLM scene planner with common providers.

    Supported providers:
        - "transformers": Local Hugging Face pipeline (default model: "Qwen/Qwen2.5-1.5B-Instruct", <12GB VRAM).
        - "rule_based" / "offline": Built-in deterministic semantic planner (no GPU or API keys required).
        - "openai": OpenAI ChatCompletion (e.g. model="gpt-4o", model="gpt-4o-mini").
        - "gemini": Google Gemini API (e.g. model="gemini-1.5-flash").
        - "auto": Defaults to local transformers model, with graceful fallback to rule_based.
    """
    provider_lower = provider.lower()

    if provider_lower in ("rule_based", "offline"):
        return RuleBasedPlanner(creative=creative)

    if provider_lower in ("transformers", "auto"):
        m = model or "Qwen/Qwen2.5-1.5B-Instruct"
        try:
            return TransformersPlanner(
                model_name=m,
                creative=creative,
                fallback_planner=RuleBasedPlanner(creative=creative),
                max_retries=max_retries,
            )
        except Exception as e:
            logger.warning(f"Could not initialize TransformersPlanner: {e}. Falling back to RuleBasedPlanner.")
            return RuleBasedPlanner(creative=creative)

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
                fallback_planner=RuleBasedPlanner(creative=creative),
                creative=creative,
                max_retries=max_retries,
            )
        except ImportError:
            return RuleBasedPlanner(creative=creative)

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
                fallback_planner=RuleBasedPlanner(creative=creative),
                creative=creative,
                max_retries=max_retries,
            )
        except ImportError:
            return RuleBasedPlanner(creative=creative)

    return RuleBasedPlanner(creative=creative)
