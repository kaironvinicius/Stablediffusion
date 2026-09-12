"""Model presets, device selection and filesystem layout."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Weights and the Hugging Face cache live inside the repo so that a container
# with a mounted workspace keeps them across runs. Both are gitignored.
MODELS_DIR = Path(os.environ.get("SDCHAT_MODELS_DIR", REPO_ROOT / "models"))
OUTPUT_DIR = Path(os.environ.get("SDCHAT_OUTPUT_DIR", REPO_ROOT / "outputs"))


@dataclass(frozen=True)
class Preset:
    """A named model configuration with defaults tuned for its architecture."""

    repo_id: str
    steps: int
    guidance: float
    size: int
    # Turbo models are distilled to run without classifier-free guidance, which
    # also means they ignore negative prompts.
    supports_negative: bool


PRESETS: dict[str, Preset] = {
    "sd-turbo": Preset(
        repo_id="stabilityai/sd-turbo",
        steps=2,
        guidance=0.0,
        size=512,
        supports_negative=False,
    ),
    "sdxl-turbo": Preset(
        repo_id="stabilityai/sdxl-turbo",
        steps=2,
        guidance=0.0,
        size=512,
        supports_negative=False,
    ),
    "sd15": Preset(
        repo_id="runwayml/stable-diffusion-v1-5",
        steps=25,
        guidance=7.5,
        size=512,
        supports_negative=True,
    ),
    "sdxl": Preset(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        steps=30,
        guidance=7.0,
        size=1024,
        supports_negative=True,
    ),
}

DEFAULT_PRESET = "sd-turbo"


def pick_device() -> str:
    """Return the best available torch device."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    """Half precision pays off on GPU; on CPU it is slower than float32."""
    import torch

    return torch.float32 if device == "cpu" else torch.float16


def preset_for(model: str) -> Preset:
    """Map a --model value to the defaults that suit it.

    A preset name uses its own tuning. Anything else is a custom repo id or a
    local checkpoint: assume a conventional (non-distilled) model, because
    turbo defaults of two steps at zero guidance would render noise.
    """
    if model in PRESETS:
        return PRESETS[model]
    lowered = model.lower()
    if "turbo" in lowered or "lightning" in lowered:
        return PRESETS["sd-turbo"]
    if "xl" in lowered:
        return PRESETS["sdxl"]
    return PRESETS["sd15"]
