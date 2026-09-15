"""Tests for authentication, 9-area grid & corners DSL, SAM3 override, and search strategy."""

import json
import os
import numpy as np
import pytest
from pathlib import Path
from PIL import Image
import cv2

from image_paster.auth import (
    save_auth_token,
    get_stored_token,
    clear_auth_token,
    get_api_key,
    get_auth_status,
    mask_token,
    AUTH_CONFIG_FILE,
)
from image_paster.cli import main
from image_paster.dsl import parse_dsl, SceneIR
from image_paster.scene.layout import SemanticLayoutSolver
from image_paster.segmentation.sam3 import SAM3Segmenter
from image_paster.pipeline.generator import SemanticImageGenerator
from image_paster.retrieval.mock import MockRetriever
from image_paster.llm.planner import RuleBasedPlanner, GeminiScenePlanner, create_llm_planner


# ---------------------------------------------------------------------------
# 1. Auth Management Tests
# ---------------------------------------------------------------------------

def test_auth_save_and_retrieve(tmp_path, monkeypatch):
    test_auth_file = tmp_path / "auth.json"
    monkeypatch.setattr("image_paster.auth.AUTH_CONFIG_FILE", test_auth_file)
    monkeypatch.setattr("image_paster.auth.AUTH_CONFIG_DIR", tmp_path)

    # Empty initially
    assert get_stored_token() is None

    # Save token
    saved = save_auth_token("AIzaSyFakeTestToken12345678")
    assert saved == test_auth_file
    assert test_auth_file.exists()

    # Retrieve token
    token = get_stored_token()
    assert token == "AIzaSyFakeTestToken12345678"

    # Mask token
    masked = mask_token(token)
    assert masked == "AIza...5678"

    # Clear token
    assert clear_auth_token() is True
    assert get_stored_token() is None
    assert not test_auth_file.exists()


def test_auth_api_key_resolution_order(tmp_path, monkeypatch):
    test_auth_file = tmp_path / "auth.json"
    monkeypatch.setattr("image_paster.auth.AUTH_CONFIG_FILE", test_auth_file)
    monkeypatch.setattr("image_paster.auth.AUTH_CONFIG_DIR", tmp_path)

    # 1. Explicit key has highest precedence
    monkeypatch.setenv("GOOGLE_API_KEY", "env_google_key")
    monkeypatch.setenv("GEMINI_API_KEY", "env_gemini_key")
    save_auth_token("stored_key")
    assert get_api_key("explicit_key") == "explicit_key"

    # 2. GOOGLE_API_KEY environment variable takes precedence over stored
    assert get_api_key() == "env_google_key"

    # 3. GEMINI_API_KEY if GOOGLE_API_KEY unset
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert get_api_key() == "env_gemini_key"

    # 4. Stored token if env vars unset
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert get_api_key() == "stored_key"

    # 5. None if everything cleared
    clear_auth_token()
    assert get_api_key() is None


def test_cli_auth_subcommands(tmp_path, monkeypatch):
    test_auth_file = tmp_path / "auth.json"
    monkeypatch.setattr("image_paster.auth.AUTH_CONFIG_FILE", test_auth_file)
    monkeypatch.setattr("image_paster.auth.AUTH_CONFIG_DIR", tmp_path)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Status before login
    ret_status = main(["auth", "status"])
    assert ret_status == 0

    # Login with token
    ret_login = main(["auth", "login", "--token", "AIzaSyCLITestKeyABCXYZ123"])
    assert ret_login == 0
    assert test_auth_file.exists()
    assert get_stored_token() == "AIzaSyCLITestKeyABCXYZ123"

    # Status after login
    ret_status2 = main(["auth", "status"])
    assert ret_status2 == 0

    # Logout
    ret_logout = main(["auth", "logout"])
    assert ret_logout == 0
    assert not test_auth_file.exists()


# ---------------------------------------------------------------------------
# 2. DSL 9-Area Grid & Corners Tests
# ---------------------------------------------------------------------------

def test_dsl_nine_grid_areas_parsing_and_layout():
    dsl = """
    scene NineAreaGridScene {
        camera { viewpoint = eye_level; perspective = natural; }
        environment { search("neutral studio backdrop"); type = "studio"; }
        objects {
            object obj_tl { source { search("obj1"); } region = top_left; }
            object obj_tc { source { search("obj2"); } region = top_center; }
            object obj_tr { source { search("obj3"); } region = top_right; }
            object obj_cl { source { search("obj4"); } region = center_left; }
            object obj_cc { source { search("obj5"); } region = center; }
            object obj_cr { source { search("obj6"); } region = center_right; }
            object obj_bl { source { search("obj7"); } region = bottom_left; }
            object obj_bc { source { search("obj8"); } region = bottom_center; }
            object obj_br { source { search("obj9"); } region = bottom_right; }
        }
    }
    """
    ir = parse_dsl(dsl, validate=True)
    assert len(ir.objects) == 9

    solver = SemanticLayoutSolver(canvas_width=1000, canvas_height=1000)
    layout = solver.solve(ir)

    # Verify 3 columns: left (~200), center (~500), right (~800)
    assert layout.objects["obj_tl"].center[0] == pytest.approx(200, abs=30)
    assert layout.objects["obj_tc"].center[0] == pytest.approx(500, abs=30)
    assert layout.objects["obj_tr"].center[0] == pytest.approx(800, abs=30)

    assert layout.objects["obj_cl"].center[0] == pytest.approx(200, abs=30)
    assert layout.objects["obj_cc"].center[0] == pytest.approx(500, abs=30)
    assert layout.objects["obj_cr"].center[0] == pytest.approx(800, abs=30)

    assert layout.objects["obj_bl"].center[0] == pytest.approx(200, abs=30)
    assert layout.objects["obj_bc"].center[0] == pytest.approx(500, abs=30)
    assert layout.objects["obj_br"].center[0] == pytest.approx(800, abs=30)

    # Verify 3 rows: top (<400), center (~500), bottom (>600)
    assert layout.objects["obj_tl"].center[1] < 400
    assert layout.objects["obj_tc"].center[1] < 400
    assert layout.objects["obj_tr"].center[1] < 400

    assert 400 <= layout.objects["obj_cl"].center[1] <= 600
    assert 400 <= layout.objects["obj_cc"].center[1] <= 600
    assert 400 <= layout.objects["obj_cr"].center[1] <= 600

    assert layout.objects["obj_bl"].center[1] > 600
    assert layout.objects["obj_bc"].center[1] > 600
    assert layout.objects["obj_br"].center[1] > 600


def test_dsl_corners_parsing():
    dsl = """
    scene CornerObjectsScene {
        camera { viewpoint = eye_level; }
        environment { search("gallery wall"); type = "room"; }
        objects {
            object c1 { source { search("lamp"); } region = corner_top_left; }
            object c2 { source { search("clock"); } region = corner_top_right; }
            object c3 { source { search("plant"); } region = corner_bottom_left; }
            object c4 { source { search("box"); } region = corner_bottom_right; }
        }
    }
    """
    ir = parse_dsl(dsl, validate=True)
    assert ir.objects["c1"].region == "corner_top_left"
    assert ir.objects["c2"].region == "corner_top_right"
    assert ir.objects["c3"].region == "corner_bottom_left"
    assert ir.objects["c4"].region == "corner_bottom_right"

    solver = SemanticLayoutSolver(canvas_width=1000, canvas_height=1000)
    layout = solver.solve(ir)
    assert layout.objects["c1"].center[0] < 300 and layout.objects["c1"].center[1] < 400
    assert layout.objects["c2"].center[0] > 700 and layout.objects["c2"].center[1] < 400
    assert layout.objects["c3"].center[0] < 300 and layout.objects["c3"].center[1] > 600
    assert layout.objects["c4"].center[0] > 700 and layout.objects["c4"].center[1] > 600


# ---------------------------------------------------------------------------
# 3. DSL Override / Replacement Syntax Tests
# ---------------------------------------------------------------------------

def test_dsl_override_syntaxes():
    # Syntax 1: replaces property assignment
    dsl1 = """
    scene S1 {
        environment { search("forest with a person standing"); }
        objects {
            object monkey {
                source { search("monkey full body"); }
                replaces = "human";
                region = center;
            }
        }
    }
    """
    ir1 = parse_dsl(dsl1, validate=True)
    assert ir1.objects["monkey"].replaces == "human"

    # Syntax 2: object = override("target")
    dsl2 = """
    scene S2 {
        environment { search("forest with a person standing"); }
        objects {
            object monkey = override("human") {
                source { search("monkey full body"); }
                region = center;
            }
        }
    }
    """
    ir2 = parse_dsl(dsl2, validate=True)
    assert ir2.objects["monkey"].replaces == "human"

    # Syntax 3: monkey.replaces(human) in relations
    dsl3 = """
    scene S3 {
        environment { search("forest with a person standing"); }
        objects {
            object monkey {
                source { search("monkey full body"); }
                region = center;
            }
        }
        relations {
            monkey.replaces(human);
        }
    }
    """
    ir3 = parse_dsl(dsl3, validate=True)
    assert ir3.objects["monkey"].replaces == "human"

    # Test round-trip C++ DSL output
    cpp_out = ir1.to_cpp_dsl()
    assert 'replaces = "human";' in cpp_out or 'replaces(human)' in cpp_out
    ir1_roundtrip = parse_dsl(cpp_out, validate=True)
    assert ir1_roundtrip.objects["monkey"].replaces == "human"


# ---------------------------------------------------------------------------
# 4. SAM3 Background Detection & Image Override Pipeline Tests
# ---------------------------------------------------------------------------

def test_sam3_detect_in_background_contour_fallback():
    segmenter = SAM3Segmenter(force_fallback=True)

    # Create a background with a salient white rectangle (representing a person)
    bg = np.zeros((600, 800, 3), dtype=np.uint8)
    # Background texture
    bg[:] = (40, 50, 40)
    # Target object in center
    cv2.rectangle(bg, (350, 180), (450, 420), (220, 220, 220), -1)

    det_res = segmenter.detect_in_background(bg, prompt="person")
    assert det_res is not None
    assert not det_res.rejected
    assert det_res.area > 0

    x, y, w, h = det_res.bbox
    # Bounding box should overlap with the rectangle at (350, 180, 100, 240)
    assert 300 <= x <= 400
    assert 150 <= y <= 250


def test_generator_override_vs_fallback_trace(tmp_path):
    segmenter = SAM3Segmenter(force_fallback=True)
    retriever = MockRetriever()

    # Case A: Background HAS the target object
    bg_with_human = np.zeros((600, 800, 3), dtype=np.uint8)
    bg_with_human[:] = (35, 45, 35)
    cv2.rectangle(bg_with_human, (350, 180), (450, 420), (230, 230, 230), -1)

    gen = SemanticImageGenerator(retriever=retriever, segmenter=segmenter)

    dsl_override = """
    scene ReplaceHumanScene {
        environment { search("forest with a person standing"); }
        objects {
            object monkey {
                source { search("monkey"); }
                replaces = "human";
                region = bottom_right;
            }
        }
    }
    """

    res_detected = gen.generate(
        prompt="monkey in forest replacing human",
        dsl_override=dsl_override,
        background_image=bg_with_human,
    )
    trace_detected = res_detected.execution_trace

    assert "overrides" in trace_detected
    assert "monkey" in trace_detected["overrides"]
    assert trace_detected["overrides"]["monkey"]["status"] == "DETECTED_AND_OVERRIDDEN"
    assert "bbox" in trace_detected["overrides"]["monkey"]

    # Verify monkey layout was placed near the detected human box (x~400) rather than bottom_right (x~800)
    monkey_layout = res_detected.layout_plan.objects["monkey"]
    assert 300 <= monkey_layout.center[0] <= 500

    # Case B: Background is flat/empty (Target NOT detected)
    bg_empty = np.zeros((600, 800, 3), dtype=np.uint8)
    bg_empty[:] = (35, 45, 35)

    res_not_detected = gen.generate(
        prompt="monkey in forest replacing human",
        dsl_override=dsl_override,
        background_image=bg_empty,
    )
    trace_not_detected = res_not_detected.execution_trace

    assert "overrides" in trace_not_detected
    assert "monkey" in trace_not_detected["overrides"]
    assert trace_not_detected["overrides"]["monkey"]["status"] == "NOT_DETECTED_FALLBACK_NORMAL"

    # Verify monkey layout falls back to normal region (bottom_right -> x ~ 80% = 819)
    monkey_fallback_layout = res_not_detected.layout_plan.objects["monkey"]
    assert monkey_fallback_layout.center[0] > 700


# ---------------------------------------------------------------------------
# 5. Search Strategy for Replacement
# ---------------------------------------------------------------------------

def test_replacement_search_strategy():
    planner = RuleBasedPlanner()
    dsl_text, ir = planner.plan("a monkey in a misty pine forest replacing a human")

    assert "with a person standing" in dsl_text
    assert ir.environment.query and "with a person standing" in ir.environment.query
    assert "monkey" in ir.objects
    assert ir.objects["monkey"].replaces in ("human", "person")


def test_create_llm_planner_defaults_to_gemini():
    planner = create_llm_planner()
    assert isinstance(planner, GeminiScenePlanner)
    assert planner.primary_model_name == "gemini-2.5-flash"
