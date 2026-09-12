#!/usr/bin/env bash
# Pre-download a model's weights so the first generation is not the slow one.
#
#   ./scripts/fetch_model.sh            # sd-turbo, the CPU default
#   ./scripts/fetch_model.sh sdxl-turbo
set -euo pipefail

PRESET="${1:-sd-turbo}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

HF_HOME="$ROOT/models/hf" "$PY" - "$PRESET" <<'PYEOF'
import sys

from sdchat.config import PRESETS

name = sys.argv[1]
if name not in PRESETS:
    sys.exit(f"unknown preset {name!r}; choose one of: {', '.join(sorted(PRESETS))}")

repo_id = PRESETS[name].repo_id
print(f"downloading {repo_id} ...")

from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id,
    allow_patterns=["*.json", "*.txt", "*.safetensors"],
    ignore_patterns=["*.onnx*", "*.msgpack", "*.bin"],
)
print(f"cached at {path}")
PYEOF
