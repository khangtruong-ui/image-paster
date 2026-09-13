"""Unit tests for the end-to-end Semantic Image Generator pipeline."""

import json
from pathlib import Path
from image_paster.pipeline.generator import SemanticImageGenerator
from image_paster.retrieval.mock import MockRetriever
from image_paster.segmentation.sam3 import SAM3Segmenter


def test_end_to_end_generation(tmp_path):
    generator = SemanticImageGenerator(
        retriever=MockRetriever(cache_dir=tmp_path / "cache"),
        segmenter=SAM3Segmenter(force_fallback=True),
    )

    prompt = "an elephant standing behind a tree in a forest"
    result = generator.generate(prompt=prompt, blend_mode="poisson")

    assert result.image_rgb.shape == (1024, 1024, 3)
    assert len(result.dsl_text) > 50
    assert "elephant" in result.scene_ir.objects
    assert "tree" in result.scene_ir.objects

    # Verify trace content
    trace = result.execution_trace
    assert trace["prompt"] == prompt
    assert "planning" in trace["pipeline_stages"]
    assert "retrieval" in trace["pipeline_stages"]
    assert "segmentation" in trace["pipeline_stages"]
    assert "elephant" in trace["retrieval"]
    assert "elephant" in trace["segmentation"]
    assert "layout" in trace
    assert "rendering" in trace
    assert "verification" in trace

    # Test saving
    out_img = tmp_path / "test.png"
    out_dsl = tmp_path / "test.dsl"
    out_tr = tmp_path / "test_trace.json"
    result.save(out_img, out_dsl, out_tr)

    assert out_img.exists()
    assert out_dsl.exists()
    assert out_tr.exists()

    with open(out_tr) as f:
        saved_trace = json.load(f)
    assert saved_trace["prompt"] == prompt


def test_generator_with_dsl_override(tmp_path):
    dsl_content = """
    scene DirectScene {
        objects {
            object car {
                depth = foreground;
                region = center;
                standing_on = ground;
            }
        }
        relations {
            car.standing_on(ground);
        }
    }
    """
    generator = SemanticImageGenerator(
        retriever=MockRetriever(cache_dir=tmp_path / "cache"),
        segmenter=SAM3Segmenter(force_fallback=True),
    )

    result = generator.generate(prompt="dummy", dsl_override=dsl_content, blend_mode="alpha")
    assert result.scene_ir.name == "DirectScene"
    assert "car" in result.scene_ir.objects
    assert result.execution_trace["planner"] == "override"


def test_debug_mode_artifacts(tmp_path):
    dbg_dir = tmp_path / "debug_test"
    generator = SemanticImageGenerator(
        retriever=MockRetriever(cache_dir=tmp_path / "cache"),
        segmenter=SAM3Segmenter(force_fallback=True),
        debug=True,
    )

    result = generator.generate(
        prompt="an elephant standing behind a tree in a forest",
        debug=True,
        debug_dir=dbg_dir,
    )

    assert (dbg_dir / "00_compiled_scene.dsl").exists()
    assert (dbg_dir / "01_retrieval_elephant_1.png").exists()
    assert (dbg_dir / "02_segmentation_elephant_mask.png").exists()
    assert (dbg_dir / "02_segmentation_elephant_cutout.png").exists()
    assert (dbg_dir / "03_layout_wireframe.png").exists()
    assert (dbg_dir / "04_composite_alpha.png").exists()
    assert (dbg_dir / "04_composite_poisson.png").exists()
    assert (dbg_dir / "05_composite_final.png").exists()
    assert (dbg_dir / "debug_summary.json").exists()
