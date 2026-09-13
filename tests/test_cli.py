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
