"""Unit tests for Scene Graph, Depth Solver, Constraint Solver, and Layout Solver."""

import pytest
from image_paster.dsl import parse_dsl
from image_paster.scene.graph import SceneGraph
from image_paster.scene.depth import DepthSolver, DepthResolutionError
from image_paster.scene.constraints import ConstraintSolver, ConstraintViolation
from image_paster.scene.layout import SemanticLayoutSolver


def test_scene_graph_construction():
    dsl = """
    scene GraphTest {
        objects {
            object lion { depth = foreground; }
            object rock { depth = background; }
        }
        relations {
            lion.behind(rock);
        }
    }
    """
    ir = parse_dsl(dsl)
    sg = SceneGraph.from_scene_ir(ir)
    assert "lion" in sg.nodes
    assert "rock" in sg.nodes
    assert "ground" in sg.nodes
    outgoing = sg.get_outgoing_edges("lion")
    assert any(e.name == "behind" and e.target == "rock" for e in outgoing)


def test_depth_solver_relaxation():
    dsl = """
    scene DepthTest {
        objects {
            object a { depth = foreground; }
            object b { depth = foreground; }
        }
        relations {
            a.behind(b); // Requires a < b despite both starting at foreground
        }
    }
    """
    ir = parse_dsl(dsl)
    solver = DepthSolver()
    depths = solver.solve(ir)
    assert depths["a"][0] < depths["b"][0]
    assert depths["a"][1] < depths["b"][1]


def test_constraint_solver_evaluations():
    boxes = {
        "cat": (100, 200, 80, 80),   # bottom = 280
        "box": (80, 280, 120, 100),  # top = 280
    }
    z_indices = {"cat": 1, "box": 0}

    # cat touching box top should pass
    from image_paster.dsl.ir import ConstraintIR
    c_touch = ConstraintIR(subject="cat", constraint="must_touch", target="box")
    violations = ConstraintSolver.evaluate(
        constraints=[c_touch],
        boxes=boxes,
        z_indices=z_indices,
        ground_y=500,
    )
    assert len(violations) == 0


def test_layout_solver_derives_coordinates():
    dsl = """
    scene LayoutTest {
        camera {
            viewpoint = eye_level;
        }
        objects {
            object elephant {
                depth = foreground;
                region = center;
                standing_on = ground;
                transformation { scale = large; }
            }
        }
        relations {
            elephant.standing_on(ground);
        }
        constraints {
            elephant.must_touch(ground);
        }
    }
    """
    ir = parse_dsl(dsl)
    solver = SemanticLayoutSolver(canvas_width=1000, canvas_height=1000)
    layout_plan = solver.solve(ir, extracted_sizes={"elephant": (400, 300)})

    assert layout_plan.canvas_width == 1000
    assert layout_plan.canvas_height == 1000
    el = layout_plan.objects["elephant"]
    assert el.width > 0
    assert el.height > 0
    # Bottom should touch ground
    assert abs((el.y + el.height) - layout_plan.ground_y) <= 5
