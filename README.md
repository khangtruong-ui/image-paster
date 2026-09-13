# Image Paster: Diffusion-Free Semantic Image Generation

A research-oriented Python framework for generating composite scenes from natural-language prompts **without using diffusion models**.

Instead of delegating the entire creative process to a black-box diffusion generator, this project implements a **semantic scene programming pipeline**:

```text
User Natural Language Prompt
            ↓
  LLM Scene Planner (Qwen2.5-1.5B-Instruct / Transformers / Rule-Based)
            ↓
   C++ style Scene DSL
            ↓
Parser & Semantic Validator (Lark)
            ↓
     Scene IR & Graph
            ↓
Image Retrieval (DuckDuckGo Search for Foreground Objects & Background)
            ↓
  Object Segmentation (SAM 3 / jetjodh/sam3 Mirror)
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

1. **Default Local Transformers LLM Planner (<12GB VRAM GPU)**:
   - Utilizes **`Qwen/Qwen2.5-1.5B-Instruct`** from Hugging Face's `transformers` as the default scene planner.
   - Optimized for single GPUs with **<12GB VRAM** using `float16` precision (~2.9GB VRAM footprint). Coexists effortlessly on consumer/workstation GPUs alongside SAM 3 (~1.7GB VRAM, total <5GB VRAM).
   - Pluggable planner architecture supporting `transformers`, deterministic offline `rule_based`, `openai`, `gemini`, or `auto`.
   - Automatic graceful fallback to `rule_based` if offline or if model weights are unavailable.

2. **Photographic Background Retrieval via `search("...")`**:
   - The environment block supports explicit search declarations: `environment { search("dense misty pine forest landscape"); type = "forest"; ... }`.
   - DuckDuckGo retrieves high-resolution, real-world landscape and setting photographs to use as the base canvas, replacing procedural 2-color sky/ground gradients with authentic photographic backgrounds.

3. **Creative Mode with Contextual Decorative Accents (Default)**:
   - By default, the planner operates in **Creative Mode**: in addition to the primary subjects requested in the prompt, it automatically introduces 1–2 small contextual decorative elements (e.g. wildflowers, shrubs, rocks, street lamps, potted plants) appropriate for the detected setting.
   - Can be strictly disabled at any time with the **`--prompt-only`** flag to generate only the explicit entities specified in the prompt.

4. **C++ Style Semantic Scene DSL**:
   - Structured grammar with C++ syntax: blocks (`{ ... }`), statements terminated by semicolons (`;`), object definitions, method-style relations (`elephant.behind(tree);`), declarative constraints (`tree.must_occlude(elephant);`), and C++ single-line and multi-line comments (`//`, `/* ... */`).
   - Parsed with industrial-strength parsing tools (**Lark** with Earley context-free grammar).
   - Rich error diagnostics reporting line and column positions.

5. **SAM 3 Object Segmentation & Mirror Fallback**:
   - Object extraction using **Segment Anything Model 3 (SAM 3)** from Hugging Face's `transformers` (`Sam3Model` and `Sam3Processor`).
   - Supports primary repository `facebook/sam3` and automatic seamless fallback to the mirror repository **`jetjodh/sam3`** if gated access is not pre-approved.
   - Configurable via `--sam3-model`, `--sam3-mirror`, and `--hf-token`.
   - Automated candidate evaluation and rejection (checks foreground ratio, boundary compactness, and confidence thresholds).
   - Automatic graceful CV-based fallback mode for offline testing and environments without model weights.

6. **DuckDuckGo Image Retrieval with Explicit Search Prompts**:
   - Synthesizes object-specific search queries from semantic source requirements or direct search calls (e.g. `elephant = search("red africa elephant");` or `source { search("red africa elephant"); }`).
   - Caches images and records complete provenance metadata (URL, source, ranking, timestamp).
   - Includes mock/synthetic retriever for deterministic offline reproduction.

7. **Semantic Scene & Layout Solver**:
   - Compiles semantic spatial relationships (`standing_on`, `left_of`, `behind`, `near`) into numerical 2D canvas coordinates without requiring the LLM to guess pixel coordinates.
   - Depth order resolution and topological relaxation for `distant`, `background`, `midground`, and `foreground`.
   - Perspective projection scaling and ground plane anchoring.

8. **OpenCV Compositing & Poisson Seamless Blending**:
   - Geometric transformations: perspective warp, rotation, horizontal/vertical flipping, scaling.
   - Pixel-accurate occlusion calculation and mask intersection.
   - Poisson seamless cloning (`cv2.seamlessClone`) with boundary safety guards preventing OpenCV assertion crashes, with alpha-feathered fallback.
   - Illumination and color temperature matching (`warm`, `cool`, brightness, contrast).

9. **Visual & Semantic Verification**:
   - Verifies whether rendered scenes satisfy the semantic DSL constraints (ground contact, occlusion, visibility, relative depth).
   - Returns structured `PASS` or `FAIL` reports with concrete remediation suggestions.
   - Bounded automatic retry loop for self-correcting scene generation.

10. **Research-Oriented Execution Trace**:
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

### 1. Generate an Image from a Natural Language Prompt

```bash
# Standard generation (uses Qwen2.5-1.5B, DuckDuckGo retrieval, SAM 3, and creative accents)
image-paster generate "an elephant standing behind a tree in a forest"

# Prompt-only mode (strictly prompt entities, disabling creative decorative accents)
image-paster generate "an elephant standing behind a tree in a forest" --prompt-only

# Offline mode (uses synthetic mock retriever and deterministic rule-based planner)
image-paster generate "an elephant standing behind a tree in a forest" --offline --output output.png

# Verbose debug mode (saves all stage artifacts and intermediate visualizations)
image-paster generate "a red panda sitting on a wooden chair inside a spaceship" --debug --debug-dir debug/
```

### 2. Compile a Prompt to C++ Scene DSL

```bash
# Compile prompt to C++ Scene DSL using default transformers model
image-paster plan "a vintage car parked near a cafe"

# Compile with prompt-only mode (no decorative accents)
image-paster plan "a vintage car parked near a cafe" --prompt-only

# Compile using rule-based offline planner
image-paster plan "an elephant in a forest" --llm-provider rule_based
```

### 3. Parse and Validate a Scene DSL File

```bash
image-paster parse examples/elephant_in_forest.dsl
```

---

## CLI Flags Reference

The `image-paster` command-line tool provides full control over all stages of the generation pipeline.

### `image-paster generate`

```text
usage: image-paster generate [prompt] [options]
```

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `prompt` | `str` | `""` | Natural language scene prompt (optional if `--dsl` is provided). |
| `--dsl` | `path` | `None` | Path to an existing C++ Scene DSL file to bypass LLM planning. |
| `--output`, `-o` | `path` | `output.png` | Destination path for the generated composite image. |
| `--trace`, `-t` | `path` | `execution_trace.json` | Destination path for the execution trace JSON report. |
| `--dsl-out` | `path` | `generated_scene.dsl` | Destination path for the compiled C++ Scene DSL code. |
| `--blend` | `choice` | `natural` | Compositing blend mode: `natural`, `alpha`, or `poisson`. |
| `--prompt-only` | `flag` | `False` | Disables creative additions; generates strictly prompt-specified entities. |
| `--llm-provider` | `choice` | `transformers` | Scene planner engine: `transformers`, `rule_based`, `openai`, `gemini`, `auto`. |
| `--llm-model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Model identifier for transformers or API planners (<12GB VRAM default). |
| `--offline` | `flag` | `False` | Forces offline mode (synthetic mock retriever and rule-based planner). |
| `--sam3-model` | `str` | `facebook/sam3` | Primary Hugging Face repository for SAM 3 segmentation. |
| `--sam3-mirror`| `str` | `jetjodh/sam3` | Fallback mirror repository for SAM 3 (un-gated). |
| `--hf-token` | `str` | `None` | Hugging Face authentication token for gated model access. |
| `--debug` | `flag` | `False` | Enables verbose console logging and saves intermediate stage artifacts. |
| `--debug-dir` | `path` | `debug` | Directory path where debug artifacts are written. |

### `image-paster plan`

```text
usage: image-paster plan <prompt> [options]
```

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `prompt` | `str` | *required* | Natural language prompt to compile. |
| `--prompt-only` | `flag` | `False` | Disables creative mode and decorative object generation. |
| `--llm-provider` | `choice` | `transformers` | Planner engine: `transformers`, `rule_based`, `openai`, `gemini`, `auto`. |
| `--llm-model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Model identifier to use for compilation. |

### `image-paster parse`

```text
usage: image-paster parse <dsl_file>
```

| Argument | Type | Description |
| :--- | :--- | :--- |
| `dsl_file` | `path` | Path to a C++ Scene DSL file to parse and semantically validate. |

---

## Creative Mode vs. `--prompt-only`

Real-world images look sparse and artificial when only main foreground actors are pasted onto a blank surface. To create rich, grounded scenes, `image-paster` includes **Creative Mode**:

- **Creative Mode (Default)**:
  The scene planner analyzes the scene's setting and automatically adds 1–2 small, contextual decorative objects on the ground or background (e.g., a cluster of wildflowers in a forest, sea shells on a beach, a small cactus in a desert, or a street lamp in a city).
  These decorative objects are assigned `scale = small`, positioned away from the primary subject to avoid occlusion, and anchored to the ground plane.

- **Prompt-Only Mode (`--prompt-only`)**:
  When `--prompt-only` is passed, creative additions are disabled. The planner outputs strictly the entities explicitly mentioned in the user prompt.

```bash
# Adds contextual wildflowers in addition to elephant and tree
image-paster generate "an elephant standing behind a tree in a forest"

# Only generates elephant and tree
image-paster generate "an elephant standing behind a tree in a forest" --prompt-only
```

---

## Photographic Background Retrieval via `search("...")`

The C++ Scene DSL environment block supports direct background search queries:

```cpp
    environment {
        search("dense lush green pine forest landscape photo");
        type = "forest";
        ground = "grassy";
        lighting {
            direction = upper_left;
            intensity = medium;
            temperature = warm;
        }
    }
```

When present:
1. The retrieval engine searches DuckDuckGo for the background photograph.
2. The image is downloaded, verified, and supplied as the canvas backdrop.
3. In `--debug` mode, the downloaded background is preserved as `01_retrieval_background.png`.
4. Downstream OpenCV compositing and Poisson seamless blending blend the segmented objects directly into the real background photograph.

---

## Local Transformers LLM (<12GB VRAM)

By default, `image-paster` uses **`Qwen/Qwen2.5-1.5B-Instruct`**:
- **VRAM Footprint**: ~2.9GB VRAM in `torch.float16`.
- **Coexistence**: Runs concurrently with SAM 3 (~1.7GB VRAM) on a single 6GB, 8GB, 12GB, or 16GB GPU with >7GB headroom remaining.
- **Latency**: ~1.5 to 2.5 seconds per scene plan on modern GPUs.
- **Reliability**: Generates valid C++ Scene DSL with automated retries and fallback to `RuleBasedPlanner` if offline.

```python
from image_paster.llm import create_llm_planner, TransformersPlanner

# Instantiate default transformers planner
planner = create_llm_planner(provider="transformers", model="Qwen/Qwen2.5-1.5B-Instruct")
dsl_text, scene_ir = planner.plan("a cat resting on a wooden chair")
```

---

## SAM 3 Model & Mirror Fallback (`jetjodh/sam3`)

The segmentation stage utilizes Meta's **Segment Anything Model 3 (SAM 3)**:

1. **Primary (`facebook/sam3`)**: Loaded by default using local Hugging Face cache or `--hf-token` / `HF_TOKEN`.
2. **Mirror Repository (`jetjodh/sam3`)**: If the primary checkpoint requires manual gated access approval that has not yet been authorized, the system automatically falls back to `jetjodh/sam3` (an un-gated mirror).
3. **Computer Vision Heuristic Fallback**: If offline or if neither remote repository can be reached, the system automatically falls back to OpenCV GrabCut and alpha thresholding.

```bash
# Explicitly use mirror repository
image-paster generate "a red africa elephant in a savanna" --sam3-model jetjodh/sam3

# Provide Hugging Face token for gated models
image-paster generate "an elephant in a forest" --hf-token "hf_..."
```

---

## Debug Mode & Stage Inspection

Enable Debug Mode using `--debug` in the CLI or `debug=True` in Python:

```bash
image-paster generate "an elephant standing behind a tree in a forest" --debug --debug-dir debug/
```

### Intermediate Visual Artifacts

Debug Mode automatically exports step-by-step visual artifacts to the `debug/` directory:

| Filename | Description |
| :--- | :--- |
| `00_compiled_scene.dsl` | Compiled and validated C++ Scene DSL code |
| `01_retrieval_background.png` | Real photographic background downloaded from DuckDuckGo |
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

`image-paster` provides three blending options via `--blend`:
- **`natural` (Default)**: Feathered alpha compositing that guarantees objects retain their authentic colors, luminance, and contrast while providing smooth edge transitions.
- **`alpha`**: Direct subpixel alpha compositing with opacity control.
- **`poisson`**: OpenCV seamless cloning (`cv2.seamlessClone`) with automatic luminance washout protection and boundary safety clamping.

---

## Python API Usage

```python
from image_paster import SemanticImageGenerator, create_llm_planner

# Initialize generator with transformers planner and creative mode
generator = SemanticImageGenerator(
    planner=create_llm_planner(provider="transformers", creative=True),
    creative=True,
    debug=True,
)

# Run pipeline
result = generator.generate(
    "an elephant standing behind a tree in a forest",
    blend_mode="natural",
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

## Running Tests

Run the complete test suite with `pytest`:

```bash
python3 -m pytest -v
```

All test cases covering DSL parsing, semantic validation, AST/IR roundtrips, LLM planning, environment search, creative mode, DuckDuckGo retrieval, SAM 3 mirror fallback, layout solving, OpenCV compositing, Poisson blending, visual verification, pipeline execution, and CLI flags run and pass.

---

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
