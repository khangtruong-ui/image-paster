"""Example script demonstrating programmatic usage of image-paster."""

from pathlib import Path
from image_paster import (
    SemanticImageGenerator,
    parse_dsl,
    MockRetriever,
    SAM3Segmenter,
)


def main():
    print("=== Image Paster: Diffusion-Free Semantic Image Generation ===")

    dsl_file = Path(__file__).parent / "elephant_in_forest.dsl"
    print(f"Reading DSL from {dsl_file}...")
    dsl_text = dsl_file.read_text(encoding="utf-8")

    # Parse and validate
    scene_ir = parse_dsl(dsl_text, validate=True)
    print(f"Parsed Scene IR: {scene_ir.name}")
    print(f"  Objects: {list(scene_ir.objects.keys())}")
    print(f"  Relations count: {len(scene_ir.relations)}")
    print(f"  Constraints count: {len(scene_ir.constraints)}")

    # Initialize generator in reproducible mode
    generator = SemanticImageGenerator(
        retriever=MockRetriever(),  # Use mock retriever for reproducible local run
        segmenter=SAM3Segmenter(),
    )

    print("\nRunning generation pipeline...")
    result = generator.generate(
        prompt="an elephant standing behind a tree in a forest",
        dsl_override=dsl_text,
        blend_mode="poisson",
    )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    result.save(
        image_path=output_dir / "elephant_forest.png",
        dsl_path=output_dir / "elephant_forest.dsl",
        trace_path=output_dir / "elephant_forest_trace.json",
    )

    print("\nPipeline completed successfully!")
    print(f"Image saved: {result.output_image_path}")
    print(f"DSL saved: {result.output_dsl_path}")
    print(f"Trace saved: {result.output_trace_path}")
    print(f"Verification report:\n{result.verification.format_report()}")


if __name__ == "__main__":
    main()
