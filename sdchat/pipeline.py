"""Loading and running a Stable Diffusion text-to-image pipeline."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .config import (
    DEFAULT_PRESET,
    MODELS_DIR,
    OUTPUT_DIR,
    PRESETS,
    Preset,
    pick_device,
    pick_dtype,
)

# Keep the Hugging Face cache inside the repo rather than ~/.cache, so the
# weights survive alongside the checkout.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))


class ModelUnavailable(RuntimeError):
    """Weights could not be found locally and could not be downloaded."""


@dataclass
class Result:
    path: Path
    seed: int
    steps: int
    guidance: float
    size: int
    device: str
    seconds: float


def load_pipeline(model: str = DEFAULT_PRESET, device: str | None = None):
    """Build a diffusers pipeline from a preset name or a local weights file.

    `model` is either a key of PRESETS, a Hugging Face repo id, or a path to a
    single-file `.safetensors`/`.ckpt` checkpoint.
    """
    import torch
    from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline

    device = device or pick_device()
    dtype = pick_dtype(device)

    local_file = Path(model).expanduser()
    if local_file.suffix in {".safetensors", ".ckpt"} and local_file.is_file():
        pipe = StableDiffusionPipeline.from_single_file(local_file, torch_dtype=dtype)
    else:
        repo_id = PRESETS[model].repo_id if model in PRESETS else model
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                repo_id, torch_dtype=dtype, variant=None, safety_checker=None
            )
        except Exception as exc:  # network blocked, gated repo, bad id
            raise ModelUnavailable(
                f"could not load {repo_id!r}: {exc}\n"
                "If huggingface.co is unreachable, download the weights elsewhere "
                "and pass the local path with --model."
            ) from exc

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if device == "cpu":
        # 15 GB of RAM is plenty, but slicing keeps peak usage flat and costs
        # nothing measurable when the bottleneck is CPU matmul anyway.
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        torch.set_num_threads(os.cpu_count() or 4)

    return pipe


def _slugify(prompt: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
    return slug[:limit] or "image"


def generate(
    pipe,
    prompt: str,
    *,
    preset: Preset,
    negative: str | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    size: int | None = None,
    seed: int | None = None,
    outdir: Path | None = None,
) -> Result:
    """Run one text-to-image pass and write the PNG to disk."""
    import torch

    steps = steps if steps is not None else preset.steps
    guidance = guidance if guidance is not None else preset.guidance
    size = size if size is not None else preset.size
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")

    device = str(pipe.device)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    kwargs = dict(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=size,
        height=size,
        generator=generator,
    )
    # Passing a negative prompt at guidance 0 is silently ignored by diffusers,
    # so only forward it where it actually influences the result.
    if negative and preset.supports_negative and guidance > 1.0:
        kwargs["negative_prompt"] = negative

    started = time.monotonic()
    image = pipe(**kwargs).images[0]
    elapsed = time.monotonic() - started

    outdir = outdir or OUTPUT_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{int(time.time())}-{seed}-{_slugify(prompt)}.png"
    image.save(path)

    return Result(
        path=path,
        seed=seed,
        steps=steps,
        guidance=guidance,
        size=size,
        device=device,
        seconds=elapsed,
    )
