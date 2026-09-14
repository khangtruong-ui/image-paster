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


def test_cli_adjust_to_file(tmp_path):
    example_path = Path("examples/elephant_in_forest.dsl")
    out_dsl = tmp_path / "adjusted.dsl"
    ret = main([
        "adjust",
        str(example_path),
        "make the elephant larger and shift it higher",
        "-o", str(out_dsl),
        "--llm-provider", "rule_based",
    ])
    assert ret == 0
    assert out_dsl.exists()
    content = out_dsl.read_text(encoding="utf-8")
    assert "elephant" in content
    assert "scale" in content


def test_cli_adjust_stdout():
    example_path = Path("examples/elephant_in_forest.dsl")
    ret = main([
        "adjust",
        str(example_path),
        "put it on the left",
        "--llm-provider", "rule_based",
    ])
    assert ret == 0


def test_cli_generate_with_threshold_flags(tmp_path):
    out_img = str(tmp_path / "thresh_out.png")
    out_dsl = str(tmp_path / "thresh_out.dsl")
    ret = main([
        "generate",
        "a car on a road",
        "--offline",
        "--max-area-ratio", "0.90",
        "--min-area-ratio", "0.02",
        "--output", out_img,
        "--dsl-out", out_dsl,
    ])
    assert ret == 0
    assert Path(out_img).exists()


