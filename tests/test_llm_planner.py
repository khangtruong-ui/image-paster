"""Unit tests for LLM Scene Planner and prompt management."""

from image_paster.llm.planner import (
    RuleBasedPlanner,
    LLMScenePlanner,
    extract_dsl_from_response,
)
from image_paster.llm.prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES


def test_rule_based_planner():
    planner = RuleBasedPlanner()
    prompt = "an elephant standing behind a tree in a forest"
    dsl_text, ir = planner.plan(prompt)

    assert "Elephant" in ir.name or "Scene" in ir.name
    assert "forest" in ir.environment.env_type
    assert "elephant" in ir.objects
    assert "tree" in ir.objects
    assert any(r.subject == "elephant" and r.relation == "behind" and r.target == "tree" for r in ir.relations)


def test_extract_dsl_from_response():
    markdown_response = """
    Here is your requested scene:
    ```cpp
    scene TestExtract {
        objects {
            object lion { depth = foreground; }
        }
    }
    ```
    Enjoy!
    """
    extracted = extract_dsl_from_response(markdown_response)
    assert extracted.startswith("scene TestExtract")
    assert extracted.endswith("}")


def test_mock_llm_planner_with_callback():
    def mock_llm(sys_prompt, user_prompt):
        return """
        ```cpp
        scene PlannedScene {
            objects {
                object car {
                    depth = foreground;
                    standing_on = ground;
                }
            }
            relations {
                car.standing_on(ground);
            }
        }
        ```
        """

    planner = LLMScenePlanner(llm_fn=mock_llm)
    dsl_text, ir = planner.plan("a red car")
    assert ir.name == "PlannedScene"
    assert "car" in ir.objects


def test_prompts_integrity():
    assert len(SYSTEM_PROMPT) > 100
    assert len(FEW_SHOT_EXAMPLES) >= 2
    assert "prompt" in FEW_SHOT_EXAMPLES[0]
    assert "dsl" in FEW_SHOT_EXAMPLES[0]


def test_create_llm_planner_factory():
    from image_paster.llm.planner import create_llm_planner, TransformersPlanner
    planner = create_llm_planner("offline")
    assert isinstance(planner, RuleBasedPlanner)
    auto_planner = create_llm_planner("auto")
    assert auto_planner is not None


def test_creative_mode_vs_prompt_only():
    prompt = "an elephant standing behind a tree in a forest"

    # Creative mode (default: creative=True)
    planner_creative = RuleBasedPlanner(creative=True)
    _, ir_creative = planner_creative.plan(prompt)
    assert "elephant" in ir_creative.objects
    assert "tree" in ir_creative.objects
    # Should include a decorative object (e.g. wildflowers)
    assert len(ir_creative.objects) >= 3

    # Prompt-only mode (creative=False)
    planner_prompt_only = RuleBasedPlanner(creative=False)
    _, ir_prompt_only = planner_prompt_only.plan(prompt)
    assert "elephant" in ir_prompt_only.objects
    assert "tree" in ir_prompt_only.objects
    assert len(ir_prompt_only.objects) == 2


def test_environment_search_in_planner():
    planner = RuleBasedPlanner()
    prompt = "a lion resting in a savanna"
    dsl_text, ir = planner.plan(prompt)

    assert ir.environment.query is not None
    assert len(ir.environment.query) > 0
    assert 'search("' in dsl_text


def test_transformers_planner_fallback():
    from image_paster.llm.planner import TransformersPlanner
    planner = TransformersPlanner(model_name="nonexistent/model_that_does_not_exist_xyz")
    dsl_text, ir = planner.plan("a red panda on a chair in a room")

    # Should gracefully fall back to RuleBasedPlanner
    assert ir is not None
    assert len(ir.objects) > 0
    assert any(k in ir.objects for k in ("red", "panda", "red_panda", "subject"))


def test_transformers_planner_ladder():
    from image_paster.llm.planner import TransformersPlanner, DEFAULT_2_TO_5B_MODELS
    # Default ladder
    planner = TransformersPlanner()
    assert planner.model_candidates == DEFAULT_2_TO_5B_MODELS
    assert "google/gemma-4-E2B" in planner.model_candidates[0]

    # Custom user model prepended
    custom_planner = TransformersPlanner(model_name="custom/my-model-4b")
    assert custom_planner.model_candidates[0] == "custom/my-model-4b"
    assert "google/gemma-4-E2B" in custom_planner.model_candidates
    assert "Qwen/Qwen3.5-2B" in custom_planner.model_candidates


def test_natural_search_query_generation():
    planner = RuleBasedPlanner()
    prompt = "a vintage convertible car on a coastal road"
    dsl_text, ir = planner.plan(prompt)

    # Environment query should be natural and clean
    assert len(ir.environment.query) > 0
    assert not any(w in ir.environment.query.lower() for w in ("8k", "dslr", "studio lighting"))

    # Object query should be concise and natural without bloating
    car_obj = next(iter(ir.objects.values()))
    assert "car on the road" in car_obj.source.query.lower() or "car" in car_obj.source.query.lower()
    assert not any(w in car_obj.source.query.lower() for w in ("8k", "dslr", "studio lighting", "plain white background"))


