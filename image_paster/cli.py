"""Command-line interface for image-paster."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from image_paster.dsl import parse_dsl, DSLSyntaxError, DSLValidationError
from image_paster.llm.planner import create_llm_planner
from image_paster.retrieval.mock import MockRetriever
from image_paster.segmentation.sam3 import SAM3Segmenter
from image_paster.pipeline.generator import SemanticImageGenerator
from image_paster.auth import (
    save_auth_token,
    get_auth_status,
    clear_auth_token,
    mask_token,
)


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="image-paster",
        description="Diffusion-Free Semantic Image Generation using C++ Scene DSL and SAM 3",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 0. auth
    auth_parser = subparsers.add_parser("auth", help="Manage Google AI Studio / Gemini authentication")
    auth_subparsers = auth_parser.add_subparsers(dest="auth_command", required=True)

    login_parser = auth_subparsers.add_parser("login", help="Authenticate with Google AI Studio API key")
    login_parser.add_argument("--token", "-t", type=str, default=None, help="Google AI Studio API key (e.g. $GOOGLE_API_KEY)")

    status_parser = auth_subparsers.add_parser("status", help="Check current authentication status")

    logout_parser = auth_subparsers.add_parser("logout", help="Log out and clear stored API token")

    # 1. generate
    gen_parser = subparsers.add_parser("generate", help="Generate an image from prompt or DSL")
    gen_parser.add_argument("prompt", type=str, nargs="?", default="", help="Natural language prompt")
    gen_parser.add_argument("--dsl", type=str, default=None, help="Path to input C++ Scene DSL file")
    gen_parser.add_argument("--output", "-o", type=str, default="output.png", help="Output image path")
    gen_parser.add_argument("--trace", "-t", type=str, default="execution_trace.json", help="Execution trace JSON path")
    gen_parser.add_argument("--dsl-out", type=str, default="generated_scene.dsl", help="Saved DSL file path")
    gen_parser.add_argument("--blend", choices=["natural", "alpha", "poisson"], default="natural", help="Blending mode (natural, alpha, or poisson)")
    gen_parser.add_argument("--offline", action="store_true", help="Force offline mode using synthetic mock retriever and rule-based planner")
    gen_parser.add_argument("--prompt-only", action="store_true", help="Disable creative decorative additions; generate strictly prompt-specified entities")
    gen_parser.add_argument("--llm-provider", choices=["gemini", "transformers", "rule_based", "openai", "auto"], default="gemini", help="LLM planner provider (default: gemini)")
    gen_parser.add_argument("--llm-model", type=str, default="gemini-2.5-flash", help="LLM model identifier or comma-separated fallback ladder (default: gemini-2.5-flash)")
    gen_parser.add_argument("--api-key", type=str, default=None, help="Google AI Studio / Gemini API key")
    gen_parser.add_argument("--sam3-model", type=str, default="facebook/sam3", help="Primary Hugging Face repository for SAM 3 (default: facebook/sam3)")
    gen_parser.add_argument("--sam3-mirror", type=str, default="jetjodh/sam3", help="Fallback mirror repository for SAM 3 (default: jetjodh/sam3)")
    gen_parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face authentication token for gated model access")
    gen_parser.add_argument("--max-area-ratio", type=float, default=0.95, help="Upper threshold on segmentation mask area ratio (default: 0.95)")
    gen_parser.add_argument("--min-area-ratio", type=float, default=0.01, help="Lower threshold on segmentation mask area ratio (default: 0.01)")
    gen_parser.add_argument("--debug", action="store_true", help="Enable verbose debug mode and save stage artifacts")
    gen_parser.add_argument("--debug-dir", type=str, default="debug", help="Directory to save debug stage artifacts")

    # 2. parse
    parse_parser = subparsers.add_parser("parse", help="Parse and validate a C++ Scene DSL file")
    parse_parser.add_argument("dsl_file", type=str, help="Path to C++ Scene DSL file")

    # 3. plan
    plan_parser = subparsers.add_parser("plan", help="Compile a natural language prompt into C++ Scene DSL")
    plan_parser.add_argument("prompt", type=str, help="Natural language prompt")
    plan_parser.add_argument("--prompt-only", action="store_true", help="Disable creative mode and generate only explicitly mentioned objects")
    plan_parser.add_argument("--llm-provider", choices=["gemini", "transformers", "rule_based", "openai", "auto"], default="gemini", help="LLM planner provider (default: gemini)")
    plan_parser.add_argument("--llm-model", type=str, default="gemini-2.5-flash", help="LLM model identifier or comma-separated fallback ladder (default: gemini-2.5-flash)")
    plan_parser.add_argument("--api-key", type=str, default=None, help="Google AI Studio / Gemini API key")
    plan_parser.add_argument("--debug", action="store_true", help="Enable verbose debug mode and print full raw LLM reasoning")

    # 4. adjust
    adjust_parser = subparsers.add_parser("adjust", help="Adjust an existing compiled C++ Scene DSL file based on a new prompt")
    adjust_parser.add_argument("dsl_file", type=str, help="Path to input C++ Scene DSL file")
    adjust_parser.add_argument("prompt", type=str, help="Adjustment instruction (e.g., 'The object A is too low, put it a little higher')")
    adjust_parser.add_argument("--output", "-o", type=str, default=None, help="Output path for adjusted DSL file (prints to stdout if omitted)")
    adjust_parser.add_argument("--llm-provider", choices=["gemini", "transformers", "rule_based", "openai", "auto"], default="gemini", help="LLM planner provider (default: gemini)")
    adjust_parser.add_argument("--llm-model", type=str, default="gemini-2.5-flash", help="LLM model identifier or comma-separated fallback ladder")
    adjust_parser.add_argument("--api-key", type=str, default=None, help="Google AI Studio / Gemini API key")
    adjust_parser.add_argument("--debug", action="store_true", help="Enable verbose debug mode")

    # 5. render (generate directly from a DSL file)
    render_parser = subparsers.add_parser("render", help="Generate/render an image directly from a C++ Scene DSL file")
    render_parser.add_argument("dsl_file", type=str, help="Path to input C++ Scene DSL file")
    render_parser.add_argument("--output", "-o", type=str, default="output.png", help="Output image path")
    render_parser.add_argument("--trace", "-t", type=str, default="execution_trace.json", help="Execution trace JSON path")
    render_parser.add_argument("--blend", choices=["natural", "alpha", "poisson"], default="natural", help="Blending mode (natural, alpha, or poisson)")
    render_parser.add_argument("--offline", action="store_true", help="Force offline mode using synthetic mock retriever")
    render_parser.add_argument("--sam3-model", type=str, default="facebook/sam3", help="Primary Hugging Face repository for SAM 3 (default: facebook/sam3)")
    render_parser.add_argument("--sam3-mirror", type=str, default="jetjodh/sam3", help="Fallback mirror repository for SAM 3 (default: jetjodh/sam3)")
    render_parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face authentication token for gated model access")
    render_parser.add_argument("--max-area-ratio", type=float, default=0.95, help="Upper threshold on segmentation mask area ratio (default: 0.95)")
    render_parser.add_argument("--min-area-ratio", type=float, default=0.01, help="Lower threshold on segmentation mask area ratio (default: 0.01)")
    render_parser.add_argument("--debug", action="store_true", help="Enable verbose debug mode and save stage artifacts")
    render_parser.add_argument("--debug-dir", type=str, default="debug", help="Directory to save debug stage artifacts")

    parsed_args = parser.parse_args(args)

    if parsed_args.command == "auth":
        if parsed_args.auth_command == "login":
            token = parsed_args.token
            if not token:
                import os
                token = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not token:
                try:
                    import getpass
                    token = getpass.getpass("Enter Google AI Studio API key: ").strip()
                except Exception:
                    pass
            if not token:
                print("Error: No API token provided. Use --token <API_KEY> or set $GOOGLE_API_KEY.", file=sys.stderr)
                return 1

            saved_path = save_auth_token(token)
            print(f"[OK] Successfully authenticated with Google AI Studio!")
            print(f"     Token: {mask_token(token)}")
            print(f"     Saved credentials to: {saved_path}")
            return 0

        elif parsed_args.auth_command == "status":
            st = get_auth_status()
            if st["authenticated"]:
                print(f"[OK] Authenticated: Yes")
                print(f"     Source: {st['source']}")
                print(f"     Active Key: {st['masked_key']}")
            else:
                print(f"[!] Authenticated: No")
                print(f"     Run 'image-paster auth login --token <API_KEY>' or export GOOGLE_API_KEY=<KEY>")
            return 0

        elif parsed_args.auth_command == "logout":
            cleared = clear_auth_token()
            if cleared:
                print(f"[OK] Stored Google AI Studio credentials successfully removed.")
            else:
                print(f"[INFO] No stored credentials found to remove.")
            return 0

    elif parsed_args.command == "parse":
        p = Path(parsed_args.dsl_file)
        if not p.exists():
            print(f"Error: DSL file not found at {p}", file=sys.stderr)
            return 1
        content = p.read_text(encoding="utf-8")
        try:
            ir = parse_dsl(content, validate=True)
            print(f"Valid Scene DSL! Scene Name: {ir.name}")
            print(f"Objects ({len(ir.objects)}): {list(ir.objects.keys())}")
            print(f"Relations ({len(ir.relations)}): {[(r.subject, r.relation, r.target) for r in ir.relations]}")
            print(f"Constraints ({len(ir.constraints)}): {[(c.subject, c.constraint, c.target) for c in ir.constraints]}")
            return 0
        except (DSLSyntaxError, DSLValidationError) as e:
            print(f"Validation failed:\n{e}", file=sys.stderr)
            return 2

    elif parsed_args.command == "plan":
        creative = not parsed_args.prompt_only
        is_debug = getattr(parsed_args, "debug", False)
        planner = create_llm_planner(
            provider=parsed_args.llm_provider,
            model=parsed_args.llm_model,
            api_key=getattr(parsed_args, "api_key", None),
            creative=creative,
            debug=is_debug,
        )
        dsl_text, _ = planner.plan(parsed_args.prompt)
        if is_debug and getattr(planner, "last_raw_response", None):
            print(f"[DEBUG:Planning] Full LLM Reasoning (Raw Text):\n{planner.last_raw_response}\n")
        print(dsl_text)
        return 0

    elif parsed_args.command == "adjust":
        dsl_path = Path(parsed_args.dsl_file)
        if not dsl_path.exists():
            print(f"Error: DSL file not found: {dsl_path}", file=sys.stderr)
            return 1
        existing_dsl = dsl_path.read_text(encoding="utf-8")
        is_debug = getattr(parsed_args, "debug", False)
        planner = create_llm_planner(
            provider=parsed_args.llm_provider,
            model=parsed_args.llm_model,
            api_key=getattr(parsed_args, "api_key", None),
            debug=is_debug,
        )
        adjusted_dsl, _ = planner.adjust_dsl(existing_dsl, parsed_args.prompt)
        if parsed_args.output:
            out_p = Path(parsed_args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(adjusted_dsl, encoding="utf-8")
            print(f"[OK] Adjusted Scene DSL saved to: {out_p}")
        else:
            print(adjusted_dsl)
        return 0

    elif parsed_args.command == "render":
        dsl_path = Path(parsed_args.dsl_file)
        if not dsl_path.exists():
            print(f"Error: DSL file not found: {dsl_path}", file=sys.stderr)
            return 1
        dsl_content = dsl_path.read_text(encoding="utf-8")
        prompt = dsl_path.stem

        print(f"==> Rendering scene directly from DSL file: '{dsl_path}'...")
        retriever = MockRetriever() if parsed_args.offline else None
        segmenter = SAM3Segmenter(
            model_name=parsed_args.sam3_model,
            mirror_model_name=parsed_args.sam3_mirror,
            hf_token=parsed_args.hf_token,
            min_area_ratio=parsed_args.min_area_ratio,
            max_area_ratio=parsed_args.max_area_ratio,
            force_fallback=parsed_args.offline,
        )

        generator = SemanticImageGenerator(
            retriever=retriever,
            segmenter=segmenter,
            debug=parsed_args.debug,
        )

        try:
            result = generator.generate(
                prompt=prompt,
                dsl_override=dsl_content,
                blend_mode=parsed_args.blend,
                debug=parsed_args.debug,
                debug_dir=parsed_args.debug_dir,
            )
            result.save(
                image_path=parsed_args.output,
                trace_path=parsed_args.trace,
            )
            print(f"[OK] Image successfully rendered and saved to: {parsed_args.output}")
            print(f"[OK] Execution trace saved to: {parsed_args.trace}")
            print(f"Verification: {result.verification.format_report()}")
            return 0
        except Exception as e:
            print(f"[FAIL] Rendering failed: {e}", file=sys.stderr)
            return 3

    elif parsed_args.command == "generate":
        dsl_content = None
        # Auto-detect if positional prompt argument is a path to a DSL file
        if parsed_args.prompt and Path(parsed_args.prompt).is_file() and (parsed_args.prompt.endswith(".dsl") or not parsed_args.dsl):
            dsl_path = Path(parsed_args.prompt)
            dsl_content = dsl_path.read_text(encoding="utf-8")
            prompt = dsl_path.stem
        elif parsed_args.dsl:
            dsl_path = Path(parsed_args.dsl)
            if not dsl_path.exists():
                print(f"Error: DSL file not found: {dsl_path}", file=sys.stderr)
                return 1
            dsl_content = dsl_path.read_text(encoding="utf-8")
            prompt = parsed_args.prompt or dsl_path.stem
        else:
            prompt = parsed_args.prompt or "scene"

        print(f"==> Generating scene for: '{prompt}'...")

        creative = not parsed_args.prompt_only
        provider = "rule_based" if parsed_args.offline else parsed_args.llm_provider
        planner = create_llm_planner(
            provider=provider,
            model=parsed_args.llm_model,
            api_key=getattr(parsed_args, "api_key", None),
            creative=creative,
            hf_token=parsed_args.hf_token,
            debug=parsed_args.debug,
        )

        retriever = MockRetriever() if parsed_args.offline else None
        segmenter = SAM3Segmenter(
            model_name=parsed_args.sam3_model,
            mirror_model_name=parsed_args.sam3_mirror,
            hf_token=parsed_args.hf_token,
            min_area_ratio=parsed_args.min_area_ratio,
            max_area_ratio=parsed_args.max_area_ratio,
            force_fallback=parsed_args.offline,
        )

        generator = SemanticImageGenerator(
            planner=planner,
            retriever=retriever,
            segmenter=segmenter,
            creative=creative,
            debug=parsed_args.debug,
        )

        try:
            result = generator.generate(
                prompt=prompt,
                dsl_override=dsl_content,
                blend_mode=parsed_args.blend,
                debug=parsed_args.debug,
                debug_dir=parsed_args.debug_dir,
            )
            result.save(
                image_path=parsed_args.output,
                dsl_path=parsed_args.dsl_out,
                trace_path=parsed_args.trace,
            )
            print(f"[OK] Image successfully saved to: {parsed_args.output}")
            print(f"[OK] Scene DSL saved to: {parsed_args.dsl_out}")
            print(f"[OK] Execution trace saved to: {parsed_args.trace}")
            print(f"Verification: {result.verification.format_report()}")
            return 0
        except Exception as e:
            print(f"[FAIL] Generation failed: {e}", file=sys.stderr)
            return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
