# Image Paster: Diffusion-Free Semantic Image Generation

A research-oriented Python framework for generating composite scenes from natural-language prompts **without using diffusion models**.

Instead of delegating the entire creative process to a black-box diffusion generator, this project implements a **semantic scene programming pipeline**:

```text
User Natural Language Prompt
            ↓
  LLM Scene Planner (2-5B Model Ladder: Qwen2.5-3B → Qwen2.5-1.5B → Rule-Based)
            ↓
   C++ style Scene DSL
            ↓
Parser & Semantic Validator (Lark)
            ↓
     Scene IR & Graph
            ↓
Image Retrieval (DuckDuckGo Search with Complex Prompts for Objects & Background)
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

1. **Default 2–5B LLM Planner with Multi-Tier OOM Fallback (<12GB VRAM GPU)**:
   - Primary default model: **`Qwen/Qwen2.5-3B-Instruct`** (~3.09B parameters, ~5.75GB VRAM in fp16).
   - Coexists comfortably alongside SAM 3 (~1.7GB VRAM) on a single 12GB or 16GB GPU (total ~7.5GB VRAM, well within 12GB limits).
   - **Automatic OOM Fallback Ladder**: If the primary 3B model encounters a CUDA `OutOfMemoryError` during loading or generation, it immediately evacuates VRAM (`torch.cuda.empty_cache()`), falls back to **`Qwen/Qwen2.5-1.5B-Instruct`** (~1.54B parameters, ~2.9GB VRAM), and finally to deterministic `RuleBasedPlanner`.
   - Custom models easily specified via `--llm-model` (e.g. `--llm-model microsoft/Phi-3.5-mini-instruct` or comma-separated ladder).

2. **Complex & Descriptive Search Query Mandate**:
   - The system prompt strictly urges and mandates elaborate, multi-attribute photographic search queries for both foreground objects and background environments.
   - Eliminates generic queries like `search("elephant")` in favor of rich queries like `search("majestic adult African bush elephant with large ivory tusks walking forward full body isolated on clean white background studio lighting DSLR")`.
   - Results in dramatically higher quality, higher resolution image retrieval from DuckDuckGo and cleaner SAM 3 cutouts.

3. **Photographic Background Retrieval via `search("...")`**:
   - The environment block supports explicit search declarations: `environment { search("panoramic landscape photography of dense misty redwood pine forest with sunbeams 8k"); type = "forest"; ... }`.
   - DuckDuckGo retrieves high-resolution, real-world landscape and setting photographs to use as the base canvas, replacing procedural 2-color sky/ground gradients with authentic photographic backgrounds.

4. **Creative Mode with Contextual Decorative Accents (Default)**:
   - By default, the planner operates in **Creative Mode**: in addition to the primary subjects requested in the prompt, it automatically introduces 1–2 small contextual decorative elements (e.g. wildflowers, shrubs, rocks, street lamps, potted plants) appropriate for the detected setting.
   - Can be strictly disabled at any time with the **`--prompt-only`** flag to generate only the explicit entities specified in the prompt.

5. **C++ Style Semantic Scene DSL**:
   - Structured grammar with C++ syntax: blocks (`{ ... }`), statements terminated by semicolons (`;`), object definitions, method-style relations (`elephant.behind(tree);`), declarative constraints (`tree.must_occlude(elephant);`), and C++ single-line and multi-line comments (`//`, `/* ... */`).
   - Parsed with industrial-strength parsing tools (**Lark** with Earley context-free grammar).
   - Rich error diagnostics reporting line and column positions.

6. **SAM 3 Object Segmentation & Mirror Fallback**:
   - Object extraction using **Segment Anything Model 3 (SAM 3)** from Hugging Face's `transformers` (`Sam3Model` and `Sam3Processor`).
   - Supports primary repository `facebook/sam3` and automatic seamless fallback to the mirror repository **`jetjodh/sam3`** if gated access is not pre-approved.
   - Configurable via `--sam3-model`, `--sam3-mirror`, and `--hf-token`.
   - Automated candidate evaluation and rejection (checks foreground ratio, boundary compactness, and confidence thresholds).
   - Automatic graceful CV-based fallback mode for offline testing and environments without model weights.

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
# Standard generation (uses Qwen2.5-3B, DuckDuckGo retrieval, SAM 3, and creative accents)
image-paster generate "a vintage turquoise convertible car parked on a sunny ocean coastal road"

# Prompt-only mode (strictly prompt entities, disabling creative decorative accents)
image-paster generate "an elephant standing behind a tree in a forest" --prompt-only

# Specify a custom 2-5B LLM model
image-paster generate "a red panda sitting on a chair" --llm-model "microsoft/Phi-3.5-mini-instruct"

# Offline mode (uses synthetic mock retriever and deterministic rule-based planner)
image-paster generate "an elephant standing behind a tree in a forest" --offline --output output.png

# Verbose debug mode (saves all stage artifacts and intermediate visualizations)
image-paster generate "a red panda sitting on a wooden chair inside a spaceship" --debug --debug-dir debug/
```

### 2. Compile a Prompt to C++ Scene DSL

```bash
# Compile prompt to C++ Scene DSL using default 2-5B model (Qwen2.5-3B)
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

## 2–5B LLM Comparison & Fallback Ladder

To balance generation quality, strict schema compliance, and VRAM limits on `<12GB VRAM` GPUs, `image-paster` supports popular open-weights models in the 2–5B parameter range:

| Model | Parameters | Weights (fp16) | VRAM Footprint | Description / Access |
| :--- | :--- | :--- | :--- | :--- |
| **`microsoft/Phi-3.5-mini-instruct`** | 3.82B | ~7.1 GB | ~7.6 GB | Largest in 2-5B range. Strong reasoning & 128k context. Un-gated. |
| **`meta-llama/Llama-3.2-3B-Instruct`** | 3.21B | ~6.0 GB | ~6.5 GB | Meta LLaMA 3.2 compact instruction model. Requires gated Hugging Face approval. |
| **`Qwen/Qwen2.5-3B-Instruct`** *(Default)* | 3.09B | ~5.75 GB | ~6.1 GB | **Best structured DSL compiler.** Outstanding code generation & instruction following. Un-gated. |
| **`google/gemma-2-2b-it`** | 2.61B | ~5.0 GB | ~5.5 GB | Google Gemma 2 compact model. Requires gated Hugging Face approval. |
| **`Qwen/Qwen2.5-1.5B-Instruct`** *(Fallback)* | 1.54B | ~2.9 GB | ~3.0 GB | Ultra-lightweight fallback when VRAM is tight (<6GB) or if 3B OOMs. |

### Multi-Tier Automatic OOM Fallback
When running with `--llm-provider transformers` (the default):
1. The planner first attempts to load and generate with the primary model (**`Qwen/Qwen2.5-3B-Instruct`**).
2. If a `torch.cuda.OutOfMemoryError` occurs during model loading or during token generation, the planner immediately calls `torch.cuda.empty_cache()`, unloads the model, logs a warning, and falls back to **`Qwen/Qwen2.5-1.5B-Instruct`**.
3. If the 1.5B model also encounters an OOM or fails, the planner falls back to the deterministic offline **`RuleBasedPlanner`**.
4. The generation pipeline never crashes due to an LLM OOM.

### Specifying Custom Models via `--llm-model`
You can supply any Hugging Face model or a comma-separated fallback list via `--llm-model`:
```bash
# Use Phi-3.5-mini as primary, falling back to Qwen2.5-3B and Qwen2.5-1.5B on OOM
image-paster generate "a lion in the savanna" --llm-model "microsoft/Phi-3.5-mini-instruct"

# Explicit fallback ladder
image-paster generate "a lion in the savanna" --llm-model "meta-llama/Llama-3.2-3B-Instruct,Qwen/Qwen2.5-3B-Instruct"
```

---

## Complex & Descriptive Search Queries

To avoid generic, low-resolution, or clipart results from web image search, the prompt template strictly urges the model to generate rich, descriptive photographic queries:

- **Objects**:
  - *Bad*: `search("elephant");`
  - *Good*: `search("majestic adult African bush elephant with large ivory tusks walking forward full body isolated on clean white background studio lighting DSLR");`
  - *Bad*: `search("car");`
  - *Good*: `search("vintage turquoise convertible car with chrome accents and leather seats parked on road isolated on clean white background studio photography");`
- **Environment / Background**:
  - *Bad*: `search("forest");`
  - *Good*: `search("panoramic landscape photography of dense misty redwood pine forest with morning sunbeams streaming through canopy 8k high resolution");`
  - *Bad*: `search("beach");`
  - *Good*: `search("scenic wide-angle view of sunlit tropical beach with turquoise ocean water gentle waves and golden sand photography");`

---

## CLI Flags Reference

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
| `--llm-model` | `str` | `Qwen/Qwen2.5-3B-Instruct` | Model identifier or comma-separated fallback ladder (with automatic OOM fallback). |
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
| `--llm-model` | `str` | `Qwen/Qwen2.5-3B-Instruct` | Model identifier or comma-separated fallback ladder. |

### `image-paster parse`

```text
usage: image-paster parse <dsl_file>
```

| Argument | Type | Description |
| :--- | :--- | :--- |
| `dsl_file` | `path` | Path to a C++ Scene DSL file to parse and semantically validate. |

---

## Creative Mode vs. `--prompt-only`

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
        search("panoramic landscape photography of dense misty redwood pine forest with sunbeams 8k high resolution");
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

# Initialize generator with transformers planner (default Qwen2.5-3B with OOM fallback)
generator = SemanticImageGenerator(
    planner=create_llm_planner(provider="transformers", model="Qwen/Qwen2.5-3B-Instruct", creative=True),
    creative=True,
    debug=True,
)

# Run pipeline
result = generator.generate(
    "a vintage turquoise convertible car parked on a sunny ocean coastal road",
    blend_mode="natural",
)

# Save outputs
result.save(
    image_path="coastal_car.png",
    dsl_path="coastal_car.dsl",
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

All test cases covering DSL parsing, semantic validation, AST/IR roundtrips, LLM planning, multi-tier OOM fallback ladder, complex search queries, environment search, creative mode, DuckDuckGo retrieval, SAM 3 mirror fallback, layout solving, OpenCV compositing, Poisson blending, visual verification, pipeline execution, and CLI flags run and pass.

---

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
