#!/bin/bash
# Provision a cloud session: virtualenv, Python dependencies, model weights.
#
# Cloud containers are reclaimed after a period of inactivity, so every new
# session starts with an empty workspace. Without this, generating an image
# means reinstalling ~2 GB of wheels and re-downloading ~5 GB of weights by
# hand first.
#
# Safe to run repeatedly: each step checks for its own result before doing work.
set -euo pipefail

# Local checkouts keep their own environment; only cloud sessions start bare.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "creating virtualenv"
  python3 -m venv "$VENV"
fi

# Checking first keeps a warm container's startup to a second or two; pip would
# otherwise re-resolve the whole tree on every session.
if ! "$PY" -c "import torch, diffusers" >/dev/null 2>&1; then
  echo "installing dependencies"
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r "$ROOT/requirements.txt"
fi

# The weights are the slow part: roughly 5 GB on a cold container. Set
# SDCHAT_SKIP_MODEL=1 in the environment to start without them.
if [ "${SDCHAT_SKIP_MODEL:-0}" != "1" ]; then
  echo "fetching sd-turbo weights"
  "$ROOT/scripts/fetch_model.sh" sd-turbo
fi

# sdchat points HF_HOME here itself; exporting it means plain huggingface-cli
# and ad-hoc scripts share the same cache instead of filling ~/.cache too.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export HF_HOME=\"$ROOT/models/hf\"" >> "$CLAUDE_ENV_FILE"
fi

echo "ready"
