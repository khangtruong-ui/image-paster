# Image Paster: Diffusion-Free Semantic Image Generation

A research-oriented Python framework for generating composite scenes from natural-language prompts **without using diffusion models**.

Instead of delegating the entire creative process to a black-box diffusion generator, this project implements a **semantic scene programming pipeline**:

```text
User Natural Language Prompt
            ↓
    LLM Scene Planner
            ↓
  C++ style Scene DSL
            ↓
Parser & Semantic Validator (Lark)
            ↓
    Scene IR & Graph
            ↓
Image Retrieval (DuckDuckGo Search)
            ↓
 Object Segmentation (SAM 3)
            ↓
  Semantic Layout Solver
            ↓
Geometric Transformation & Compositing
            ↓
  Poisson / Seamless Blending
            ↓
 Visual / Semantic Verification
            ↓ (retry loop if needed)
Final Image + execution_trace.json
```

---

## Key Features

1. **C++ Style Semantic Scene DSL**:
   - Structured grammar with C++ syntax: blocks (`{ ... }`), statements terminated by semicolons (`;`), object definitions, method-style relations (`elephant.behind(tree);`), declarative constraints (`tree.must_occlude(elephant);`), and C++ single-line and multi-line comments (`//`, `/* ... */`).
   - Parsed with industrial-strength parsing tools (**Lark** with Earley context-free grammar).
   - Rich error diagnostics reporting line and column positions.

2. **Semantic Scene & Layout Solver**:
   - Compiles semantic spatial relationships (`standing_on`, `left_of`, `behind`, `near`) into numerical 2D canvas coordinates without requiring the LLM to guess pixel coordinates.
   - Depth order resolution and topological relaxation for `distant`, `background`, `midground`, and `foreground`.
   - Perspective projection scaling and ground plane anchoring.

3. **SAM 3 Object Segmentation**:
   - Object extraction using **Segment Anything Model 3 (SAM 3)** from Hugging Face's `transformers` (`Sam3Model`, `Sam3Processor`, `Sam3ImageProcessor`).
   - Automated candidate evaluation and rejection (checks foreground ratio, boundary compactness, and confidence thresholds).
   - Automatic graceful fallback mode for offline testing and environments without pre-authenticated access to gated checkpoints.

4. **DuckDuckGo Image Retrieval**:
   - Synthesizes object-specific search queries from semantic source requirements (`viewpoint = side; full_body = required; isolated = preferred;`).
   - Caches images and records complete provenance metadata (URL, source, ranking, timestamp).
   - Includes mock/synthetic retriever for deterministic offline reproduction.

5. **OpenCV Compositing & Poisson Seamless Blending**:
   - Geometric transformations: perspective warp, rotation, horizontal/vertical flipping, scaling.
   - Pixel-accurate occlusion calculation and mask intersection.
   - Poisson seamless cloning (`cv2.seamlessClone`) with boundary safety guards preventing OpenCV assertion crashes, with alpha-feathered fallback.
   - Illumination and color temperature matching (`warm`, `cool`, brightness, contrast).

6. **Visual & Semantic Verification**:
   - Verifies whether rendered scenes satisfy the semantic DSL constraints (ground contact, occlusion, visibility, relative depth).
   - Returns structured `PASS` or `FAIL` reports with concrete remediation suggestions.
   - Bounded automatic retry loop for self-correcting scene generation.

7. **Research-Oriented Execution Trace**:
   - Every run produces `execution_trace.json` recording the prompt, DSL, IR, retrieval queries, candidate metadata, segmentation scores, layout plan, render stats, verification report, and retry history.

---

## Installation

Install in editable mode:

```bash
git clone https://github.com/khangtruong-ui/image-paster.git
cd image-paster
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Command-Line Interface (CLI)

Generate a scene from a prompt:

```bash
# Offline mode (using mock retriever and local segmentation)
image-paster generate "an elephant standing behind a tree in a forest" --offline --output output.png

# Online mode (with DuckDuckGo image retrieval)
image-paster generate "a red panda sitting on a wooden chair inside a spaceship"
```

Compile a prompt to C++ Scene DSL:

```bash
image-paster plan "a vintage car parked near a cafe"
```

Parse and validate an existing DSL file:

```bash
image-paster parse examples/elephant_in_forest.dsl
```

### 2. Python API

```python
from image_paster import SemanticImageGenerator, parse_dsl

# End-to-end generation from prompt (with debug mode enabled)
generator = SemanticImageGenerator(debug=True)
result = generator.generate(
    "an elephant standing behind a tree in a forest",
    blend_mode="natural",  # "natural", "alpha", or "poisson"
)

# Save outputs
result.save(
    image_path="elephant_forest.png",
    dsl_path="elephant_forest.dsl",
    trace_path="execution_trace.json",
)

print(result.verification.format_report())
```

---

## Debug Mode & Stage Inspection

When working on research pipelines, visibility into each transformation stage is critical. Enable Debug Mode using `--debug` in the CLI or `debug=True` in Python:

```bash
# Enable debug mode in CLI
image-paster generate "an elephant standing behind a tree in a forest" --debug --debug-dir debug/
```

### 1. Granular Terminal Logging
Debug Mode explicitly logs what each module is doing in real-time, including:
- **Retrieval Queries**: Synthesized search terms and candidate URL / file paths.
- **Object Segmentation Tracking**: Explicitly outputs which object is currently being processed by SAM 3:
  ```text
  [DEBUG:Segmentation] Segmenting objects with SAM3Segmenter:
    --> Segmenting object: 'elephant'
        Candidate 1: area=46716px, score=0.95, bbox=(70, 110, 351, 381) -> ACCEPTED
    --> Segmenting object: 'tree'
        Candidate 1: area=56657px, score=0.95, bbox=(60, 50, 341, 391) -> ACCEPTED
  ```
- **Layout Coordinates**: Canvas position `(x, y)`, dimensions `(w, h)`, and z-index ordering for every object.
- **Verification Score**: Detailed checklist of spatial, occlusion, and ground contact checks.

### 2. Intermediate Visual Artifacts
Debug Mode automatically exports step-by-step visual artifacts to the `debug/` directory:

| Filename | Description |
| :--- | :--- |
| `00_compiled_scene.dsl` | Compiled and validated C++ Scene DSL code |
| `01_retrieval_<object>_<rank>.png` | Raw retrieved candidate images from search |
| `02_segmentation_<object>_mask.png` | Binary segmentation mask extracted by SAM 3 |
| `02_segmentation_<object>_cutout.png` | Segmented object cutout with transparent RGBA background |
| `03_layout_wireframe.png` | Diagnostic wireframe showing horizon, ground line, and labeled bounding boxes |
| `04_composite_alpha.png` | Full scene rendered with feathered alpha compositing |
| `04_composite_poisson.png` | Full scene rendered with Poisson seamless cloning |
| `05_composite_final.png` | Final composite chosen by the pipeline |
| `debug_summary.json` | Machine-readable execution trace and debug stats |

---

## Blending Modes: Natural vs. Poisson

Poisson blending (`cv2.seamlessClone`) solves a gradient Poisson equation. When an object is placed on a background with radically different lighting (e.g. dark spaceship or bright sky), Poisson blending can cause color washout or object fading.

`image-paster` provides three blending options via `--blend`:
- **`natural` (Default)**: Feathered alpha compositing that guarantees objects retain their authentic colors, luminance, and contrast while providing smooth edge transitions.
- **`alpha`**: Direct subpixel alpha compositing with opacity control.
- **`poisson`**: OpenCV seamless cloning (`cv2.seamlessClone`) with automatic luminance washout protection and boundary safety clamping.


---

## C++ Style Scene DSL Specification

Example scene definition in `examples/elephant_in_forest.dsl`:

```cpp
// Scene definition in C++ style DSL
scene ForestScene {
    camera {
        viewpoint = eye_level;
        perspective = natural;
        focus = elephant;
    }

    environment {
        type = "forest";
        sky = "soft_blue";
        ground = "grassy";
        lighting {
            direction = upper_left;
            intensity = medium;
            temperature = warm;
        }
    }

    objects {
        object elephant {
            source {
                viewpoint = side;
                full_body = required;
                isolated = preferred;
                resolution = high;
            }
            depth = midground;
            region = center;
            standing_on = ground;
            facing = right;
            appearance {
                color = "gray";
                lighting = inherit_scene;
            }
            transformation {
                scale = large;
                facing = right;
            }
        }

        object tree {
            source {
                viewpoint = frontal;
                isolated = preferred;
            }
            depth = foreground;
            region = left;
            standing_on = ground;
            transformation {
                scale = large;
            }
        }

        object flowers {
            depth = foreground;
            region = bottom;
            transformation {
                scale = small;
            }
        }
    }

    relations {
        elephant.behind(tree);
        elephant.standing_on(ground);
        tree.standing_on(ground);
        flowers.near(elephant);
    }

    constraints {
        elephant.must_touch(ground);
        tree.must_touch(ground);
        tree.must_occlude(elephant);
    }

    operations {
        retrieve;
        segment;
        solve_layout;
        compose;
        blend;
        verify;
    }
}
```

---

## Project Structure

```text
image-paster/
├── pyproject.toml              # Build config and dependencies
├── setup.py                    # Setuptools setup script
├── README.md                   # Documentation
├── image_paster/
│   ├── __init__.py             # Top-level exports
│   ├── cli.py                  # CLI implementation
│   ├── dsl/                    # C++ style Scene DSL
│   │   ├── grammar.py          # Lark EBNF grammar
│   │   ├── ast_nodes.py        # AST node dataclasses
│   │   ├── parser.py           # Parser and DSLSyntaxError
│   │   ├── validator.py        # Semantic validator & cycle detection
│   │   └── ir.py               # Scene IR & C++ code generator
│   ├── llm/                    # LLM Scene Planner
│   │   ├── planner.py          # RuleBasedPlanner & LLMScenePlanner
│   │   └── prompts/            # System prompts & few-shot examples
│   ├── retrieval/              # Image retrieval
│   │   ├── base.py             # Base retriever & metadata models
│   │   ├── duckduckgo.py       # DuckDuckGo image search
│   │   └── mock.py             # Deterministic mock retriever
│   ├── segmentation/           # Object segmentation
│   │   ├── base.py             # Segmenter base & rejection criteria
│   │   └── sam3.py             # SAM 3 (Hugging Face transformers)
│   ├── scene/                  # Scene & Layout Solver
│   │   ├── graph.py            # Directed scene graph
│   │   ├── depth.py            # Depth resolution & perspective scale
│   │   ├── constraints.py      # Declarative constraint solver
│   │   └── layout.py           # Semantic layout coordinate solver
│   ├── rendering/              # OpenCV rendering & blending
│   │   ├── transforms.py       # Geometric transformations
│   │   ├── masks.py            # Mask operations & occlusion stats
│   │   ├── blending.py         # Poisson seamless cloning & alpha blend
│   │   └── compositor.py       # Depth-ordered layer compositor
│   ├── verification/           # Scene verification
│   │   └── verifier.py         # Semantic visual verifier & report
│   └── pipeline/               # Pipeline orchestration
│       └── generator.py        # SemanticImageGenerator & trace logger
├── examples/
│   ├── elephant_in_forest.dsl  # Elephant scene example
│   ├── red_panda_chair.dsl     # Red panda spaceship scene
│   └── run_example.py          # Programmatic usage example
└── tests/                      # Comprehensive unit tests for all modules
    ├── test_cli.py
    ├── test_dsl_ir.py
    ├── test_dsl_parser.py
    ├── test_dsl_validator.py
    ├── test_llm_planner.py
    ├── test_pipeline.py
    ├── test_rendering.py
    ├── test_retrieval.py
    ├── test_scene_solver.py
    ├── test_segmentation.py
    └── test_verification.py
```

---

## Running Tests

Run the complete test suite with `pytest`:

```bash
python3 -m pytest -v
```

All 34 test cases covering DSL parsing, semantic validation, AST/IR roundtrips, LLM planning, image retrieval, SAM 3 segmentation, scene layout solving, OpenCV compositing, Poisson blending, visual verification, pipeline execution, and the CLI run and pass.

---

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
