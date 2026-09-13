"""Unit tests for the command-line interface (CLI)."""

from pathlib import Path
from image_paster.cli import main


def test_cli_parse():
    example_path = Path("examples/elephant_in_forest.dsl")
    ret = main(["parse", str(example_path)])
    assert ret == 0


def test_cli_plan():
    ret = main(["plan", "an elephant in a forest"])
    assert ret == 0


def test_cli_generate_offline(tmp_path):
    out_img = str(tmp_path / "cli_out.png")
    out_dsl = str(tmp_path / "cli_out.dsl")
    out_trace = str(tmp_path / "cli_trace.json")

    ret = main([
        "generate",
        "an elephant standing behind a tree in a forest",
        "--offline",
        "--output", out_img,
        "--dsl-out", out_dsl,
        "--trace", out_trace,
    ])
    assert ret == 0
    assert Path(out_img).exists()
    assert Path(out_dsl).exists()
    assert Path(out_trace).exists()


def test_cli_plan_prompt_only():
    ret = main([
        "plan",
        "an elephant in a forest",
        "--llm-provider", "rule_based",
        "--prompt-only",
    ])
    assert ret == 0


def test_cli_generate_prompt_only(tmp_path):
    out_img = str(tmp_path / "prompt_only.png")
    out_dsl = str(tmp_path / "prompt_only.dsl")

    ret = main([
        "generate",
        "an elephant in a forest",
        "--offline",
        "--prompt-only",
        "--output", out_img,
        "--dsl-out", out_dsl,
    ])
    assert ret == 0
    assert Path(out_img).exists()
    assert Path(out_dsl).exists()
    content = Path(out_dsl).read_text(encoding="utf-8")
    assert "elephant" in content
    # Should not include creative wildflowers in prompt-only mode
    assert "wildflowers" not in content

