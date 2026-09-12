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

| Preset | Time per image |
| --- | --- |
| `sd-turbo`, 2 steps | tens of seconds |
| `sd15`, 25 steps | several minutes |

Model loading dominates a single run. Use `--count` to amortise it across
several images.

## Network requirements

Weights come from `huggingface.co` and its CDN. In a sandboxed environment
whose egress policy blocks those hosts, `generate.py` fails with a message
telling you so. Two ways around it:

- Allow `huggingface.co` and `cdn-lfs.huggingface.co` in the network policy.
- Download a checkpoint on another machine and pass the file directly:
  `generate.py "a prompt" --model /path/to/model.safetensors`
