"""Offline end-to-end check of the load-and-generate path.

Builds a tiny Stable Diffusion pipeline with randomly initialised weights,
saves it to disk, then drives it through `load_pipeline` and `generate`. The
image it produces is noise, since the weights mean nothing. What it proves is
the plumbing: that a saved model directory loads, that the preset defaults and
seed reach the scheduler, and that a correctly sized PNG lands on disk.

Runs without network access, which is the point: the real weights come from
huggingface.co and that host is not always reachable.
"""

from __future__ import annotations

import json
import string
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from sdchat.config import ASPECTS, PRESETS, dimensions
from sdchat.pipeline import generate, load_pipeline

LATENT_DIM = 32
IMAGE_SIZE = 64


def write_tokenizer(target: Path) -> CLIPTokenizer:
    """CLIP's tokenizer needs vocab files on disk; synthesise a minimal pair.

    With no merge rules every word falls back to single characters, which is
    all this test needs from tokenisation.
    """
    target.mkdir(parents=True, exist_ok=True)
    specials = ["<|startoftext|>", "<|endoftext|>"]
    chars = list(string.ascii_lowercase + string.digits + " .,!-")
    tokens = specials + chars + [f"{c}</w>" for c in chars]
    vocab = {token: index for index, token in enumerate(tokens)}

    (target / "vocab.json").write_text(json.dumps(vocab))
    (target / "merges.txt").write_text("#version: 0.2\n")

    return CLIPTokenizer(
        vocab_file=str(target / "vocab.json"),
        merges_file=str(target / "merges.txt"),
        # Real CLIP checkpoints ship this; without it the default is an integer
        # too large for the tokenizer's padding path.
        model_max_length=77,
    )


def build_tiny_pipeline(workdir: Path) -> StableDiffusionPipeline:
    torch.manual_seed(0)
    tokenizer = write_tokenizer(workdir / "tokenizer")

    # The synthetic vocabulary tokenises poorly, which does not matter here:
    # the weights are random, so the embeddings carry no meaning either way.
    # Size the embedding table generously so no id can fall outside it.
    text_encoder = CLIPTextModel(
        CLIPTextConfig(
            vocab_size=max(len(tokenizer), 512),
            hidden_size=32,
            intermediate_size=37,
            num_hidden_layers=2,
            num_attention_heads=4,
            projection_dim=32,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )

    unet = UNet2DConditionModel(
        sample_size=LATENT_DIM,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(8, 8),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32,
        attention_head_dim=4,
        norm_num_groups=8,
    )

    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 8),
        latent_channels=4,
        norm_num_groups=8,
        sample_size=IMAGE_SIZE,
    )

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )

    return StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )


def main() -> int:
    workdir = REPO_ROOT / "outputs" / "_smoke"
    model_dir = workdir / "tiny-model"

    build_tiny_pipeline(workdir).save_pretrained(model_dir)
    print(f"built tiny model at {model_dir}")

    # The real loader, on a real model directory, with no network involved.
    pipe = load_pipeline(str(model_dir), device="cpu")
    print("load_pipeline OK")

    first = generate(
        pipe,
        "a lighthouse in a storm",
        preset=PRESETS["sd-turbo"],
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        seed=1234,
        outdir=workdir,
    )
    print(f"generated {first.path.name} in {first.seconds:.2f}s")

    failures = []
    if not first.path.is_file():
        failures.append("no PNG written")
    if (first.steps, first.guidance) != (2, 0.0):
        failures.append(f"preset defaults not applied: {first.steps}, {first.guidance}")
    if first.seed != 1234:
        failures.append(f"seed not honoured: {first.seed}")

    from PIL import Image

    with Image.open(first.path) as img:
        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
            failures.append(f"wrong size: {img.size}")

    # Every aspect must stay a legal VAE size and hold the pixel count steady.
    square = dimensions(512, "square")
    for aspect in ASPECTS:
        w, h = dimensions(512, aspect)
        if w % 8 or h % 8:
            failures.append(f"{aspect} is not a multiple of 8: {w}x{h}")
        if abs(w * h - square[0] * square[1]) / (square[0] * square[1]) > 0.02:
            failures.append(f"{aspect} changes the pixel count: {w}x{h}")
    if dimensions(512, "portrait")[0] >= dimensions(512, "portrait")[1]:
        failures.append("portrait is not taller than it is wide")

    # Same seed must reproduce the same bytes; a different seed must not.
    repeat = generate(
        pipe, "a lighthouse in a storm", preset=PRESETS["sd-turbo"],
        width=IMAGE_SIZE, height=IMAGE_SIZE, seed=1234, outdir=workdir,
    )
    other = generate(
        pipe, "a lighthouse in a storm", preset=PRESETS["sd-turbo"],
        width=IMAGE_SIZE, height=IMAGE_SIZE, seed=99, outdir=workdir,
    )
    if repeat.path.read_bytes() != first.path.read_bytes():
        failures.append("same seed did not reproduce the image")
    if other.path.read_bytes() == first.path.read_bytes():
        failures.append("different seed produced an identical image")

    if failures:
        for line in failures:
            print(f"FAIL {line}")
        return 1

    print("PASS load_pipeline, preset defaults, seed reproducibility, PNG output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
