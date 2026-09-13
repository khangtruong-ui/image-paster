"""Unit tests for Scene Verification module."""

import numpy as np
from image_paster.dsl import parse_dsl
from image_paster.scene.layout import SemanticLayoutSolver
from image_paster.rendering.compositor import CompositeResult
from image_paster.verification.verifier import SemanticVisualVerifier, VerificationResult


def test_visual_verifier_pass():
    dsl = """
    scene VerifyPass {
        objects {
            object tree { depth = background; standing_on = ground; }
            object flower { depth = foreground; standing_on = ground; }
        }
        relations {
            flower.standing_on(ground);
            tree.standing_on(ground);
        }
        constraints {
            flower.must_touch(ground);
            tree.must_touch(ground);
        }
    }
    """
    ir = parse_dsl(dsl)
    solver = SemanticLayoutSolver(canvas_width=400, canvas_height=400)
    layout = solver.solve(ir)

    # Mock composite
    comp = CompositeResult(
        image_bgr=np.zeros((400, 400, 3), dtype=np.uint8),
        image_rgb=np.zeros((400, 400, 3), dtype=np.uint8),
        object_canvas_masks={
            "tree": np.zeros((400, 400), dtype=np.uint8),
            "flower": np.zeros((400, 400), dtype=np.uint8),
        },
        occlusion_stats={
            "tree": {"visibility_ratio": 1.0, "occluded_by": {}},
            "flower": {"visibility_ratio": 1.0, "occluded_by": {}},
        },
    )

    verifier = SemanticVisualVerifier()
    res = verifier.verify(ir, layout, comp)
    assert res.passed
    assert res.score >= 0.8
    assert "PASS" in res.format_report()


def test_visual_verifier_fail_ground_contact():
    dsl = """
    scene VerifyFail {
        objects {
            object bird { depth = foreground; standing_on = ground; }
        }
        constraints {
            bird.must_touch(ground);
        }
    }
    """
    ir = parse_dsl(dsl)
    solver = SemanticLayoutSolver(canvas_width=400, canvas_height=400)
    layout = solver.solve(ir)
    # Manually displace object so it's floating high above ground
    layout.objects["bird"].y = 50

    comp = CompositeResult(
        image_bgr=np.zeros((400, 400, 3), dtype=np.uint8),
        image_rgb=np.zeros((400, 400, 3), dtype=np.uint8),
        object_canvas_masks={"bird": np.zeros((400, 400), dtype=np.uint8)},
        occlusion_stats={"bird": {"visibility_ratio": 1.0, "occluded_by": {}}},
    )

    verifier = SemanticVisualVerifier()
    res = verifier.verify(ir, layout, comp)
    assert not res.passed
    assert any("does not touch ground" in issue for issue in res.issues)
    assert "FAIL" in res.format_report()
