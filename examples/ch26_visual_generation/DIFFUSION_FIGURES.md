# Reproduce the Chapter 26 DDPM figures

These figures use Google's public, Apache-2.0-labelled DDPM checkpoint, loaded
through the open-source Diffusers library. The script pins the checkpoint
revision and loads safetensors weights without executing remote Python code.
It generates new samples; it does not download training images.

From the repository root, in an isolated Python environment:

```bash
python -m pip install -r requirements-diffusion-figures.txt
python -m examples.ch26_visual_generation.open_model_figures --output output/ch26-ddpm --device auto
python -m examples.ch26_visual_generation.score_figure --output output/ch26-ddpm
```

The default is `google/ddpm-cat-256` at revision
`82ca0d5db4a5ec6ff0e9be8d86852490bc18a3d9`, with 100 deterministic DDIM updates
per image and the three fixed seeds 20261110, 20261108, and 20261109. The
first seed was chosen for a legible teaching example; the gallery also shows
two other fixed draws, including their visible artifacts. This
256-by-256 model needs a substantial initial weight download. The alternative
`--model google/ddpm-cifar10-32` uses a smaller, 32-by-32 checkpoint. Outputs
vary between numerical devices and package versions; a fixed seed is not a
promise of bitwise agreement across platforms.

The reverse figure shows the actual current sample in its top row and the
model's current clean estimate in its bottom row. The scheduler clips clean
estimates to `[-1,1]` and recalculates the implied noise after clipping.
This implementation detail is recorded in the manifest; the unclipped
teaching equations are tested separately in the chapter examples.

The forward figure starts with the first generated image and performs all
1,000 prescribed forward transitions, drawing fresh independent Gaussian
increments. It does not reuse one noise tensor across the displayed levels.
Its selected display levels are 0, 50, 100, 200, 400, and 1000, which show
both the early, mild corruption and the nearly Gaussian endpoint.
Both kinds of panel use the same fixed display mapping, `clip(x/2+0.5,0,1)`.
It affects visualization, rather than rescaling every panel independently.

The run writes three PNG figures, raw arrays in `diffusion_arrays.npz`, and
`manifest.json` containing model and source hashes, software versions,
sampler settings, seeds, and figure hashes. Small-scale output examples are
illustrations of the process, not a quality benchmark.

The second command generates a separate, analytic Gaussian score diagram.
It needs no model download and illustrates an exact density and derivative.

Model cards:

- <https://huggingface.co/google/ddpm-cat-256>
- <https://huggingface.co/google/ddpm-cifar10-32>

