"""Command line entry point: one prompt in, one PNG path out."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ASPECTS, DEFAULT_ASPECT, DEFAULT_PRESET, PRESETS, dimensions, pick_device, preset_for
from .pipeline import ModelUnavailable, generate, load_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate.py",
        description="Generate an image from a text prompt with Stable Diffusion.",
    )
    parser.add_argument("prompt", help="what to draw")
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_PRESET,
        help=(
            f"preset ({', '.join(sorted(PRESETS))}), a Hugging Face repo id, "
            f"or a path to a .safetensors file (default: {DEFAULT_PRESET})"
        ),
    )
    parser.add_argument("-n", "--negative", help="what to avoid (ignored by turbo models)")
    parser.add_argument("-s", "--steps", type=int, help="denoising steps")
    parser.add_argument("-g", "--guidance", type=float, help="classifier-free guidance scale")
    parser.add_argument(
        "-a",
        "--aspect",
        choices=sorted(ASPECTS),
        default=DEFAULT_ASPECT,
        help=f"orientation, at a constant pixel count (default: {DEFAULT_ASPECT})",
    )
    parser.add_argument("--size", type=int, help="pixels per side before the aspect is applied")
    parser.add_argument("--width", type=int, help="exact width, overrides --aspect and --size")
    parser.add_argument("--height", type=int, help="exact height, overrides --aspect and --size")
    parser.add_argument("--seed", type=int, help="seed for a reproducible image")
    parser.add_argument("--count", type=int, default=1, help="how many images to generate")
    parser.add_argument("-o", "--outdir", type=Path, help="directory for the PNGs")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], help="override device")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    preset = preset_for(args.model)
    device = args.device or pick_device()
    if device == "cpu":
        print("no GPU detected, running on CPU (slower)", file=sys.stderr)

    width, height = dimensions(args.size or preset.size, args.aspect)
    # Explicit dimensions win, so an unusual canvas needs no new aspect name.
    width = args.width or width
    height = args.height or height
    if width % 8 or height % 8:
        print("error: width and height must be multiples of 8", file=sys.stderr)
        return 2

    try:
        pipe = load_pipeline(args.model, device=device)
    except ModelUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for _ in range(max(1, args.count)):
        result = generate(
            pipe,
            args.prompt,
            preset=preset,
            negative=args.negative,
            steps=args.steps,
            guidance=args.guidance,
            width=width,
            height=height,
            seed=args.seed,
            outdir=args.outdir,
        )
        print(
            f"{result.width}x{result.height}  {result.steps} steps  seed {result.seed}  "
            f"{result.seconds:.1f}s on {result.device}",
            file=sys.stderr,
        )
        # stdout carries only the path, so callers can pipe it.
        print(result.path)

    return 0
