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
    "Qwen/Qwen2.5-3B-Instruct",   # Primary 2-5B model (~3.09B params, ~5.75GB fp16)
    "Qwen/Qwen2.5-1.5B-Instruct", # Compact 1.5B fallback (~1.54B params, ~2.9GB fp16)
]


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

        # Elaborate, highly-descriptive search queries for background retrieval
        env_queries = {
            "forest": "panoramic landscape photography of dense misty redwood pine forest with sunbeams 8k high resolution",
            "woods": "scenic wide-angle photography of misty deep autumn woods forest landscape photo",
            "desert": "vast dramatic desert sand dunes under open sky cinematic landscape photography",
            "beach": "scenic wide-angle view of sunlit tropical beach turquoise ocean water and golden sand photography",
            "ocean": "deep blue open ocean water with gentle waves and horizon landscape photography",
            "mountain": "majestic snow-capped alpine mountain peak scenic landscape photography 8k",
            "snow": "winter snowy landscape with pine trees and fresh powder snow photography",
            "city": "bustling modern city street architecture wide-angle urban photography",
            "street": "urban street sidewalk architecture with warm ambient lighting photography",
            "room": "modern cozy living room interior with contemporary furniture interior photography",
            "kitchen": "bright modern kitchen interior with marble countertops interior photography",
            "park": "sunny green public park landscape with lush grass and trees photography",
            "garden": "vibrant blooming botanical garden with colorful flowers landscape photography",
            "spaceship": "wide-angle interior view of high-tech futuristic spaceship cockpit command bridge with glowing holographic display consoles cinematic lighting",
            "sky": "clear blue sky with soft white cumulus clouds panoramic sky photography",
            "studio": "clean minimalist seamless studio backdrop background photography",
        }
        env_query = env_queries.get(env_type, f"panoramic wide-angle landscape photography of {env_type} scenic background photo 8k")

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
                "forest": ("wildflowers", "delicate cluster of blooming wild alpine wildflowers on moss ground macro photography high resolution", "foreground", "bottom_right"),
                "woods": ("bush", "small lush green forest shrub bush isolated on clean white background photography", "foreground", "bottom_left"),
                "park": ("wildflowers", "small colorful blooming park flowers isolated on clean background photography", "foreground", "bottom_right"),
                "garden": ("potted_plant", "lush green flowering potted plant in ceramic pot isolated photography", "foreground", "bottom_right"),
                "beach": ("seashells", "collection of natural sea shells on beach sand macro photography", "foreground", "bottom_right"),
                "desert": ("small_cactus", "small green desert cactus in sandy soil isolated photography", "background", "bottom_left"),
                "mountain": ("pine_sapling", "small evergreen pine tree sapling on mountain soil isolated photography", "background", "bottom_left"),
                "snow": ("snowy_rock", "natural weathered granite rock covered with fresh snow isolated photography", "foreground", "bottom_left"),
                "city": ("street_lamp", "vintage black ornate street lamp post isolated on clean background photography", "background", "bottom_left"),
                "street": ("fire_hydrant", "classic red city fire hydrant on sidewalk isolated photography", "foreground", "bottom_left"),
                "room": ("houseplant", "vibrant indoor green potted houseplant in ceramic planter isolated photography", "background", "bottom_right"),
                "kitchen": ("fruit_bowl", "ceramic bowl filled with fresh colorful fruits isolated on white background", "background", "bottom_left"),
                "spaceship": ("terminal_panel", "compact sci-fi computer terminal console with glowing buttons isolated", "background", "bottom_left"),
            }
            c_name, c_query, c_depth, c_region = creative_presets.get(
                env_type, ("wildflowers", "delicate cluster of colorful wild blooming flowers isolated on clean background photography", "foreground", "bottom_right")
            )
            if c_name in (obj1, obj2):
                c_name, c_query, c_depth, c_region = ("pebbles", "small cluster of smooth river stones and pebbles isolated macro photography", "foreground", "bottom_left")
            if obj1_region == "right" or (obj2 and obj1_region == "center"):
                c_region = "bottom_left"
            else:
                c_region = "bottom_right"
            creative_item = (c_name, c_query, c_depth, c_region)

        # Build elaborate object search query with photography and isolation keywords
        obj1_query = f"{obj1.replace('_', ' ')} full body isolated on clean white background studio lighting DSLR photography"

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
            f"            }}",
            f"            transformation {{",
            f"                scale = {'medium' if (detected_relation == 'standing_on' and obj2) else 'large'};",
            f"                facing = right;",
            f"            }}",
            f"        }}",
        ]

        if obj2 and obj2 != obj1:
            obj2_query = f"{obj2.replace('_', ' ')} isolated on clean white background studio photography"
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
    """Local Hugging Face transformers scene planner using open-weights LLMs.

    Employs a multi-tier fallback ladder in the 2-5B parameter range for <12GB VRAM GPUs:
        1. Primary: 'Qwen/Qwen2.5-3B-Instruct' (~3.09B parameters, ~5.75GB VRAM in fp16)
        2. Fallback on OOM / error: 'Qwen/Qwen2.5-1.5B-Instruct' (~1.54B parameters, ~2.9GB VRAM in fp16)
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
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        self.creative = creative
        self.fallback_planner = fallback_planner or RuleBasedPlanner(creative=creative)
        self.max_retries = max_retries
        self._pipeline = None
        self.active_model_name: Optional[str] = None

        # Build candidate model ladder
        if model_candidates:
            self.model_candidates = list(model_candidates)
        elif model_name:
            user_models = [m.strip() for m in model_name.split(",") if m.strip()]
            self.model_candidates = user_models + [m for m in DEFAULT_2_TO_5B_MODELS if m not in user_models]
        else:
            self.model_candidates = list(DEFAULT_2_TO_5B_MODELS)

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

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device != "cpu" else None,
            low_cpu_mem_usage=True,
        )
        if device == "cpu":
            model = model.to("cpu")

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        return pipe

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
                logger.warning(
                    f"Failed to load '{cand}' (Error: {e}). "
                    f"Clearing VRAM and falling back down model ladder..."
                )
                self.model_candidates.pop(0)
                self._pipeline = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        logger.warning("All transformer model candidates exhausted. Falling back to RuleBasedPlanner.")
        return None

    def plan(self, prompt: str) -> Tuple[str, SceneIR]:
        import torch

        while True:
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
            oom_encountered = False

            for attempt in range(self.max_retries + 1):
                try:
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": current_user_prompt},
                    ]
                    output = pipe(messages, max_new_tokens=600, do_sample=False)
                    resp_text = output[0]["generated_text"][-1]["content"]
                    dsl_text = extract_dsl_from_response(resp_text)
                    scene_ir = parse_dsl(dsl_text, validate=True)
                    return dsl_text, scene_ir
                except torch.cuda.OutOfMemoryError as e:
                    logger.warning(
                        f"CUDA OutOfMemoryError during generation with '{self.active_model_name}': {e}. "
                        "Freeing VRAM and falling back to smaller model..."
                    )
                    oom_encountered = True
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
        - "transformers": Local Hugging Face pipeline with automatic multi-tier OOM fallback
                          (default: "Qwen/Qwen2.5-3B-Instruct" -> "Qwen/Qwen2.5-1.5B-Instruct").
        - "rule_based" / "offline": Built-in deterministic semantic planner (no GPU or API keys required).
        - "openai": OpenAI ChatCompletion (e.g. model="gpt-4o", model="gpt-4o-mini").
        - "gemini": Google Gemini API (e.g. model="gemini-1.5-flash").
        - "auto": Defaults to local transformers model ladder with graceful fallback.
    """
    provider_lower = provider.lower()

    if provider_lower in ("rule_based", "offline"):
        return RuleBasedPlanner(creative=creative)

    if provider_lower in ("transformers", "auto"):
        try:
            return TransformersPlanner(
                model_name=model or "Qwen/Qwen2.5-3B-Instruct",
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
