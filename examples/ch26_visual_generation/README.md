# Chapter 26. Image and Video Generation

Run the offline examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch26_visual_generation.run_examples --output output/ch26.json
python -m unittest discover -s tests -p test_visual_generation.py -v
```

## Mix a clean sample and noise

`add_noise` takes a clean sample, a supplied noise array of the same
shape, and a scalar cumulative signal coefficient `alpha_bar`.
The endpoints zero and one give the supplied noise and the clean
sample, respectively. The example uses two-by-two arrays to make
every multiplication visible.

## Guide an epsilon prediction

`guided_noise` takes unconditional and conditional predictions
with the same shape and a scalar guidance scale. Scale zero gives
the unconditional prediction. Scale one gives the conditional
prediction. Larger positive values extend the displacement from
the unconditional prediction toward the conditional prediction.

## Take a deterministic DDIM step

`ddim_step` takes a noisy sample, its predicted epsilon, the current
`alpha_bar_t`, and the next `alpha_bar_next`. The schedule follows
`0 < alpha_bar_t <= alpha_bar_next <= 1`. The function returns
`(estimated_clean, next_noisy)`. It first reconstructs the clean
estimate and then combines that estimate and the predicted epsilon
at the next noise level. The deterministic update uses no new
random draw.

The example follows `0.36 → 0.64 → 1.0` with the exact epsilon and
with one epsilon entry shifted by 0.25. It holds the supplied
prediction fixed across the two steps in each run. The difference
in the final arrays isolates how a prediction error enters this
calculation. In model sampling, a denoiser can produce a new
prediction at each step.

## Understand the boundary cases

The square roots scale amplitudes; their squares, not the amplitudes, add
to one. The fixed noise array is a teaching fixture, not a Gaussian draw.
`add_noise` permits `alpha_bar=0`; `ddim_step` cannot use zero as its current
level because estimating the clean sample divides by its square root.
Sampling toward cleaner states increases `alpha_bar`.

Holding a noise prediction fixed preserves the first clean estimate across
DDIM updates. Ending at level one removes the explicit noise term, but it
does not repair an inaccurate clean estimate. A real denoiser recomputes
its prediction from the updated state.

Guidance uses two predictions at the same state and noise level. Negative
coordinates and values above one are valid noise estimates. Batching the
two predictions does not make the second prediction computationally free.
Measure appearance, adherence, temporal behavior and variety separately.

See the [11 exercises](EXERCISES.md) and [complete solutions](../../solutions/ch26_visual_generation/README.md).

## Follow the pretrained runs

The [DDPM figure guide](DIFFUSION_FIGURES.md) covers the earlier pixel-space
checkpoint and the forward and reverse trajectories. The newer demonstrations
use the same toy railway scene. The image run changes one color word while
holding the starting latent fixed. The paired Wan runs hold the text and
sampling settings fixed while adding the red FLUX.2 still as the first-frame
condition.

- [Wan2.2 from text](media/wan-text.mp4)
- [Wan2.2 from text and an image](media/wan-image.mp4)
- [LingBot-Video from a structured caption](media/lingbot-text.mp4)
- [First-frame condition generated with FLUX.2](media/train-red-2501.png)
- [Recorded image run](media/image-run.json) and [recorded video runs](media/video-run.json)

Run the following commands from the repository root in a separate Python 3.12
environment on an Apple Silicon Mac with Metal. The scripts download public
weights at fixed revisions and verify their checksums. The video examples
also require FFmpeg with libx264. For a repeated offline run, add `--offline`
after the weights are cached.

```sh
python -m pip install -r requirements-ch26-media.txt
python -m examples.ch26_visual_generation.generate_images --output-dir output/ch26-image
```

The FLUX.2 Klein 4B run uses a distilled 4-bit MLX checkpoint. It saves four
images, five decoded states from one run, and a JSON record of the scheduler
levels and latent hashes.

The video sampler uses stable-diffusion.cpp at
[b9d7176](https://github.com/leejet/stable-diffusion.cpp/tree/b9d7176bf88fe465576e0aafbe47ba253ffcfc2c)
with its pinned ggml submodule. Apply the included patch to export the final
latent before decoding. The decoder comes from the pinned
[vga-mlx-video revision](https://github.com/VeryGoodAI/vga-mlx-video/tree/42514d9d47ef6c608889dbc48e6089e607a93cf1)
and uses the same downloaded VAE weights at float16 precision.
The sample command is intended for a scratch directory with sufficient disk
space and Xcode's Metal toolchain.

```sh
CH25_RUN_DIR="$PWD/output/ch26-runtime"
mkdir -p "$CH25_RUN_DIR"
git clone https://github.com/leejet/stable-diffusion.cpp "$CH25_RUN_DIR/sdcpp"
git -C "$CH25_RUN_DIR/sdcpp" checkout b9d7176bf88fe465576e0aafbe47ba253ffcfc2c
git -C "$CH25_RUN_DIR/sdcpp" submodule update --init ggml
git -C "$CH25_RUN_DIR/sdcpp" apply "$PWD/examples/ch26_visual_generation/export_video_latent.patch"
cmake -S "$CH25_RUN_DIR/sdcpp" -B "$CH25_RUN_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release -DSD_METAL=ON -DSD_WEBP=OFF -DSD_WEBM=OFF
cmake --build "$CH25_RUN_DIR/build" --target sd-cli --parallel 8
git clone https://github.com/VeryGoodAI/vga-mlx-video "$CH25_RUN_DIR/vga-mlx-video"
git -C "$CH25_RUN_DIR/vga-mlx-video" checkout 42514d9d47ef6c608889dbc48e6089e607a93cf1
python -m examples.ch26_visual_generation.generate_videos \
    --sd-cli "$CH25_RUN_DIR/build/bin/sd-cli" --decoder-source "$CH25_RUN_DIR/vga-mlx-video" \
    --image-dir output/ch26-image --output-dir output/ch26-video
python -m examples.ch26_visual_generation.make_generation_figures \
    --image-dir output/ch26-image --wan-dir output/ch26-video --lingbot-dir output/ch26-video \
    --output-dir output/ch26-figures
```

The logs and JSON record preserve the sampler settings, prompt text,
checkpoint revisions, and checksums. The two Wan rows use frames at
0, 1, 2, and 3 seconds. The shorter LingBot strip uses half-second intervals.
The figure builder also records a simple horizontal centroid of red pixels,
which can include red background features or carriages. Frame sequences and
MP4s may differ across hardware and software.
