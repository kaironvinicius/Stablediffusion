# Stable Diffusion text-to-image runner

A small, dependency-light wrapper around [diffusers](https://github.com/huggingface/diffusers)
that turns a text prompt into a PNG. It prints the file path on stdout, so an
agent or a shell script can generate an image and pick the result up in one step.

Defaults are tuned for **CPU-only machines**: the `sd-turbo` preset renders a
512px image in a couple of denoising steps instead of the usual twenty-five.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The PyPI `torch` wheel for Linux bundles the CUDA runtime and is roughly 2 GB.
On a machine that will never see a GPU you can halve that with the CPU-only
index:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Use

```bash
.venv/bin/python generate.py "a lighthouse in a storm, oil painting"
```

Weights download on first run and are cached under `models/`, which is
gitignored. To fetch them ahead of time:

```bash
./scripts/fetch_model.sh sd-turbo
```

Useful flags:

```bash
generate.py PROMPT
  -m, --model      preset name, a Hugging Face repo id, or a .safetensors path
  -n, --negative   what to avoid (turbo models ignore this, see below)
  -s, --steps      denoising steps
  -g, --guidance   classifier-free guidance scale
      --size       square output size in pixels
      --seed       reproduce an earlier image
      --count      generate several images from one model load
  -o, --outdir     where to write the PNGs (default: outputs/)
      --device     force cuda, mps or cpu
```

Progress and timings go to stderr; only the image path goes to stdout.

## Presets

| Preset | Steps | Guidance | Size | Notes |
| --- | --- | --- | --- | --- |
| `sd-turbo` | 2 | 0.0 | 512 | Default. The only preset with tolerable CPU latency. |
| `sdxl-turbo` | 2 | 0.0 | 512 | Better composition than `sd-turbo`, heavier weights. |
| `sd15` | 25 | 7.5 | 512 | Conventional model. Expect minutes per image on CPU. |
| `sdxl` | 30 | 7.0 | 1024 | Highest quality. Practical on a GPU only. |

Turbo models are distilled to run without classifier-free guidance. Guidance is
what makes a negative prompt mean anything, so `--negative` is ignored unless
you also raise `--guidance` above 1.0, which those models are not trained for.
Use `sd15` or `sdxl` when negative prompts matter.

Anything that is not a preset name is treated as a custom model, and the
defaults fall back to conventional settings rather than turbo ones. Pass
`--steps` and `--guidance` explicitly if your checkpoint wants something else.

## Performance

Measured on 4 vCPU with no GPU, 512px:

| Run | Time |
| --- | --- |
| `sd-turbo`, 2 steps, cold cache | 23.5s |
| `sd-turbo`, 2 steps, warm cache | 14.9s |

Loading the model accounts for most of a single run, so `--count` is much
cheaper per image than calling the script repeatedly. A conventional preset
such as `sd15` at 25 steps takes minutes per image on the same hardware.

## Network requirements

Weights come from `huggingface.co`, but the files themselves are served from
separate CDN and content-addressed-storage hosts. Allowing only the main domain
leaves downloads hanging partway through. Behind a domain allowlist, all four
entries are needed:

```text
huggingface.co
*.huggingface.co
*.hf.co
*.xethub.hf.co
```

`*.hf.co` covers the CDN, which resolves to names such as `us.aws.cdn.hf.co`.
`*.xethub.hf.co` covers Xet storage at `cas-server.xethub.hf.co`, which the
hub uses by default for large files. `HF_HUB_DISABLE_XET=1` forces the older
CDN path, but that host needs allowing too, so it is not a way around the list.

Where the policy cannot be changed, download a checkpoint elsewhere and pass
the file directly:

```bash
generate.py "a prompt" --model /path/to/model.safetensors
```

## Offline smoke test

`tests/test_smoke.py` builds a tiny pipeline with random weights, saves it,
then drives it through the real `load_pipeline` and `generate` functions:

```bash
.venv/bin/python tests/test_smoke.py
```

It needs no network and finishes in seconds. The image it writes is noise,
because the weights are random. What it checks is the plumbing: that a model
directory loads, that preset defaults and the seed reach the scheduler, that
the same seed reproduces an image and a different one does not, and that a
correctly sized PNG lands on disk.

## Cloud session provisioning

Cloud containers are reclaimed after a period of inactivity, so a new session
starts with an empty workspace: no virtualenv, no weights. `.claude/hooks/
session-start.sh` rebuilds both, and `.claude/settings.json` registers it as a
`SessionStart` hook, so it travels with the repository rather than living in
one person's environment settings.

The hook exits immediately unless `CLAUDE_CODE_REMOTE` is `true`, leaving local
checkouts to manage their own environment. Each step checks for its own result
first, so a warm container passes through in a few seconds. Set
`SDCHAT_SKIP_MODEL=1` to start a session without the weights when you only mean
to read or edit code.
