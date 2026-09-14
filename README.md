# Image Paster: Diffusion-Free Semantic Image Generation

A research-oriented Python framework for generating composite scenes from natural-language prompts **without using diffusion models**.

Instead of delegating the entire creative process to a black-box diffusion generator, this project implements a **semantic scene programming pipeline**:

```text
User Natural Language Prompt
            ↓
  LLM Scene Planner (Default: google/gemma-4-E2B → Qwen/Qwen3.5-2B → Rule-Based)
            ↓
   C++ style Scene DSL (with copy(...) & chained editing)
            ↓
Parser & Semantic Validator (Lark)
            ↓
     Scene IR & Graph
            ↓
Parallel Image Retrieval (DuckDuckGo Search with Concise Natural Prompts)
            ↓
  Object Segmentation (SAM 3 / jetjodh/sam3 Mirror with Mask Evaluation)
            ↓ (Search Feedback Loop: replans DSL & queries if retrieval/segmentation fails)
   Semantic Layout Solver
            ↓
Geometric Transformation & Compositing
            ↓
   Poisson / Seamless Blending
            ↓
  Visual / Semantic Verification
            ↓ (Layout retry loop if needed)
Final Image + execution_trace.json
```

---

## Key Features

1. **Default LLM Planner (`google/gemma-4-E2B`) with Multi-Tier Fallback**:
   - Primary default model: **`google/gemma-4-E2B`**.
   - Automatic fallback ladder: **`google/gemma-4-E2B`** → **`Qwen/Qwen3.5-2B`** → deterministic **`RuleBasedPlanner`**.
   - If an out-of-memory (OOM) error or model loading error occurs, VRAM is evacuated and the pipeline seamlessly tries the next model in the ladder.
   - Custom models can be specified via `--llm-model` (e.g. `--llm-model google/gemma-4-E2B,Qwen/Qwen3.5-2B`).

2. **Concise, Natural Search Queries**:
   - Prompts are designed for maximum search engine accuracy without keyword stuffing or bloated camera jargon.
   - For example, if the object is a "car" in a road scene, the query is simply `"a red car on the road"`, giving clean, relevant search results.

3. **Background Exclusivity & Scene Consistency**:
   - The environment/background is specified exclusively in `environment { search("..."); type = "..."; ... }` and is **never duplicated** as an object inside `objects { ... }`.
   - In creative mode, contextual decorative elements related to the scene (e.g. wildflowers, street lamps) are freely added to enrich the composition.

4. **Object Copying (`copy`) and Chained Transformations**:
   - Directly duplicate objects without network retrieval:
     `object car2 = copy(car) { region = right; facing(right); scale(0.8); }` or shorthand `car2 = copy(car);`.
   - Chained and nested edits: `car2.scale(0.8).facing(right);` or `edits { car2.scale(0.8); }`.
   - Copied objects reuse the source object's high-quality segmentation cutout and apply their own layout transformations, flipping, and scaling.

5. **Parallel DuckDuckGo Image Search & Download**:
   - All foreground objects are queried in parallel across threads (`ThreadPoolExecutor`).
   - Candidate images for each search query are downloaded concurrently to dramatically accelerate generation.

6. **Segmentation Mask Debugging & Configurable Thresholds**:
   - Configurable upper threshold (`--max-area-ratio`, default `0.95`) and lower threshold (`--min-area-ratio`, default `0.01`) reject full-frame images or tiny artifacts.
   - **Full Debug Mask Visibility**: In `--debug` mode, masks and transparent cutouts are saved for **every candidate** tested (both accepted and rejected):
     `02_segmentation_{name}_cand{rank}_{accepted|rejected}_mask.png` and `cutout.png`.
   - Detailed area ratio and rejection reasons are clearly printed to the console and recorded in `execution_trace.json`.

7. **Search Feedback Loop**:
   - When candidate retrieval yields 0 results or all candidates fail/are rejected by segmentation, the pipeline re-prompts the LLM (or rule-based replanner) with the previous DSL and specific failure reasons to adjust queries and retry retrieval (bounded by `max_retries`).

8. **C++ Style Semantic Scene DSL**:
   - Structured grammar with C++ syntax: blocks (`{ ... }`), statements terminated by semicolons (`;`), object definitions, copy calls, chained methods, declarative constraints, and comments (`//`, `/* ... */`).
   - Parsed with **Lark** (Earley parser) with detailed syntax error reporting.

9. **New CLI Command `image-paster adjust`**:
   - Adjust existing compiled DSL files using natural language instructions:
     `image-paster adjust scene.dsl "The car is too low, move it higher" -o adjusted.dsl`.

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
# Standard generation (uses google/gemma-4-E2B, parallel DuckDuckGo retrieval, and SAM 3)
image-paster generate "a vintage red car on the road"

# Adjust segmentation thresholds
image-paster generate "a vintage red car on the road" --max-area-ratio 0.90 --min-area-ratio 0.02

# Prompt-only mode (strictly prompt entities, disabling creative decorative accents)
image-paster generate "an elephant standing behind a tree in a forest" --prompt-only

# Offline mode (uses synthetic mock retriever and deterministic rule-based planner)
image-paster generate "an elephant standing behind a tree in a forest" --offline --output output.png

# Verbose debug mode (saves all candidate masks and intermediate stage artifacts)
image-paster generate "a car on a road" --debug --debug-dir debug/
```

### 2. Adjust an Existing DSL File with Natural Language

```bash
# Adjust existing DSL using natural language feedback
image-paster adjust scene.dsl "The car is too small, make it larger" -o adjusted.dsl
```

### 3. Compile a Prompt to C++ Scene DSL

```bash
# Compile prompt to C++ Scene DSL using default LLM (google/gemma-4-E2B)
image-paster plan "a vintage car on a road"

# Compile with prompt-only mode (no decorative accents)
image-paster plan "a vintage car on a road" --prompt-only
```

### 4. Parse and Validate a Scene DSL File

```bash
image-paster parse examples/elephant_in_forest.dsl
```

---

## 2–5B LLM Comparison & Fallback Ladder

## Default LLMs & Fallback Ladder

`image-paster` configures **`google/gemma-4-E2B`** as the primary default LLM planner, coupled with an automatic fallback ladder for resource efficiency:

| Model | Status | Description |
| :--- | :--- | :--- |
| **`google/gemma-4-E2B`** | Primary Default | High-performance compact model; largest default LLM in the repo. Outstanding structured C++ Scene DSL code generation. |
| **`Qwen/Qwen3.5-2B`** | Fallback | Fast, lightweight secondary instruction model for lower VRAM environments or fallback. |
| **`RuleBasedPlanner`** | Offline Fallback | Deterministic rule-based parser that executes without GPU or network access. |

### Multi-Tier Automatic OOM Fallback
When running with `--llm-provider transformers` (the default):
1. The planner first loads and generates with **`google/gemma-4-E2B`**.
2. If a `torch.cuda.OutOfMemoryError` occurs during model loading or token generation, the planner calls `torch.cuda.empty_cache()`, unloads the model, logs a warning, and falls back to **`Qwen/Qwen3.5-2B`**.
3. If the secondary model also encounters an OOM, the planner falls back to the deterministic offline **`RuleBasedPlanner`**.
4. The generation pipeline never crashes due to an LLM OOM.

---

## Natural, Concise Search Queries

Rather than stuffing queries with bloated camera or studio jargon that confuses search engines, the system prompt and few-shot examples mandate simple, natural search queries:

- **Objects**:
  - If the object is a "car" on a road: `search("a red car on the road");`
  - If the object is an "elephant": `search("an elephant in a forest");`
  - *No bloated keywords*: avoids unnecessary suffixes like `"studio lighting DSLR 8k octane render white background"` which degrade search engine accuracy.
- **Environment / Background**:
  - *Natural landscape*: `search("dense misty pine forest landscape");`
  - *Setting*: `search("sunny ocean coastal road landscape");`

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
| `--llm-model` | `str` | `google/gemma-4-E2B` | Model identifier or comma-separated fallback ladder (default: `google/gemma-4-E2B`). |
| `--offline` | `flag` | `False` | Forces offline mode (synthetic mock retriever and rule-based planner). |
| `--sam3-model` | `str` | `facebook/sam3` | Primary Hugging Face repository for SAM 3 segmentation. |
| `--sam3-mirror`| `str` | `jetjodh/sam3` | Fallback mirror repository for SAM 3 (un-gated). |
| `--hf-token` | `str` | `None` | Hugging Face authentication token for gated model access. |
| `--max-area-ratio`| `float`| `0.95` | Upper threshold on candidate segmentation mask area ratio. |
| `--min-area-ratio`| `float`| `0.01` | Lower threshold on candidate segmentation mask area ratio. |
| `--debug` | `flag` | `False` | Enables verbose debug mode, printing and saving all candidate masks. |
| `--debug-dir` | `path` | `debug` | Directory path where debug artifacts are written. |

### `image-paster adjust`

```text
usage: image-paster adjust <dsl_file> <prompt> [options]
```

| Argument / Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dsl_file` | `path` | *required* | Path to existing C++ Scene DSL file to modify. |
| `prompt` | `str` | *required* | Natural language adjustment instruction. |
| `--output`, `-o` | `path` | `None` | Path to save adjusted DSL file (prints to stdout if omitted). |
| `--llm-provider` | `choice` | `transformers` | Planner engine: `transformers`, `rule_based`, `openai`, `gemini`, `auto`. |
| `--llm-model` | `str` | `google/gemma-4-E2B` | Model identifier or comma-separated fallback ladder. |

### `image-paster plan`

```text
usage: image-paster plan <prompt> [options]
```

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `prompt` | `str` | *required* | Natural language prompt to compile. |
| `--prompt-only` | `flag` | `False` | Disables creative mode and decorative object generation. |
| `--llm-provider` | `choice` | `transformers` | Planner engine: `transformers`, `rule_based`, `openai`, `gemini`, `auto`. |
| `--llm-model` | `str` | `google/gemma-4-E2B` | Model identifier or comma-separated fallback ladder. |

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
