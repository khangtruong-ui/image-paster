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
