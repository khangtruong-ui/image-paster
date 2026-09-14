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
    # Candidate masks should also be saved in debug mode
    assert any(dbg_dir.glob("02_segmentation_elephant_cand*_mask.png"))
    assert (dbg_dir / "03_layout_wireframe.png").exists()
    assert (dbg_dir / "04_composite_alpha.png").exists()
    assert (dbg_dir / "04_composite_poisson.png").exists()
    assert (dbg_dir / "05_composite_final.png").exists()
    assert (dbg_dir / "debug_summary.json").exists()


def test_pipeline_copy_object(tmp_path):
    dsl_content = """
    scene CopyTestScene {
        camera {
            viewpoint = eye_level;
            perspective = natural;
            focus = car;
        }
        environment {
            search("city street");
            type = "city";
            ground = "asphalt";
        }
        objects {
            object car {
                source {
                    search("red car");
                    viewpoint = side;
                }
                depth = foreground;
                region = left;
                standing_on = ground;
            }
            object car2 = copy(car) {
                region = right;
                facing(right);
                scale(0.8);
            }
        }
        relations {
            car.standing_on(ground);
            car2.standing_on(ground);
        }
    }
    """
    generator = SemanticImageGenerator(
        retriever=MockRetriever(cache_dir=tmp_path / "cache"),
        segmenter=SAM3Segmenter(force_fallback=True),
    )

    result = generator.generate(prompt="two cars on a road", dsl_override=dsl_content)
    assert "car" in result.scene_ir.objects
    assert "car2" in result.scene_ir.objects
    assert result.scene_ir.objects["car2"].copied_from == "car"
    assert result.scene_ir.objects["car2"].transformation.scale == 0.8

    # Ensure retrieval skipped search for car2
    assert result.execution_trace["retrieval"]["car2"]["query"] == "copy(car)"
    assert result.execution_trace["retrieval"]["car2"]["candidates"] == []

    # Ensure segmentation trace records copy
    assert result.execution_trace["segmentation"]["car2"]["copied_from"] == "car"
    assert result.execution_trace["segmentation"]["car2"]["segmenter"] == "copied"


def test_pipeline_search_feedback_loop(tmp_path):
    """Test that when retrieval/segmentation fails, search feedback loop triggers replanner."""
    from image_paster.retrieval.base import RetrievalResult, ImageCandidate

    class FailingRetriever(MockRetriever):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.call_count = 0

        def retrieve_batch(self, requests, max_workers=4):
            self.call_count += 1
            results = {}
            for req in requests:
                name = req["object_name"]
                query = req["source_reqs"].query if req.get("source_reqs") else name
                if self.call_count == 1:
                    # Return 0 candidates on attempt 1
                    results[name] = RetrievalResult(object_name=name, query=query, candidates=[])
                else:
                    # Succeed on attempt 2
                    results[name] = self.retrieve(name, req.get("source_reqs"), max_results=2)
            return results

    retriever = FailingRetriever(cache_dir=tmp_path / "cache")
    generator = SemanticImageGenerator(
        retriever=retriever,
        segmenter=SAM3Segmenter(force_fallback=True),
        max_retries=2,
    )

    result = generator.generate(prompt="a car on a road")
    assert retriever.call_count > 1
    assert len(result.execution_trace["search_retry_history"]) >= 1
    assert "failures" in result.execution_trace["search_retry_history"][0]

