# Chapter 21 model demonstrations

Run the commands from the repository root. The figure renderer uses the saved
observations and the two credited public media files. Install FFmpeg with
`ffmpeg` and `ffprobe` on your path to decode the frames.

```sh
python -m pip install -r requirements.txt Pillow==13.0.0 matplotlib==3.11.2
python -m examples.ch21_vision.pretrained_figures
```

To repeat the model run, use Python 3.11 or newer and install the optional
dependencies. The first run fetches about 1.59 GB of Safetensors weights.
The runner checks the weight and media hashes. It uses the packaged model
classes, float32 on CPU, and eight Torch threads. Both commands accept
`--media-dir` and `--offline` for a repeat with cached inputs.

```sh
python -m pip install -r requirements-ch21-figures.txt
python -m examples.ch21_vision.pretrained_observations --output output/ch21-pretrained.json
python -m examples.ch21_vision.pretrained_figures --record output/ch21-pretrained.json
```

The record contains the exact input hashes, grid sizes, measured maps, frame
numbers and times, logits, and checkpoint revisions. The figure script checks
that the decoded images match the record before drawing.
[Measured observations](PRETRAINED_RUN.json).

## Trace an image patch

Follow grid position `(3, 7)` through a 32-pixel horizontal shift.
The initial projection stays equal at `(3, 9)` because those RGB values
are identical. The final descriptors differ. Cosine search with the
original descriptor retrieves `(3, 9)` as the best match in the shift.
The coordinate sums in the record use the actual first projection filter.
[Pinned DINOv3 ViT-S/16 checkpoint](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m/tree/3bf4720a82ec2066db88137180ff1f83a675cef0).

Photo by NASA, portrait of Eileen Collins, via scikit-image, public domain.
The figures resize it, translate it, and overlay a patch grid.
[Photo and attribution](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut).

## Change the order of a clip

Three manually selected windows from a single recording each contribute
16 sampled frames. The model sees each window in both orders.
The control scores each frame repeated twice and averages those logits.
With only folding and unfolding as options, the whole clip gets 6/6 views
and 3/3 pairs, and the control gets 3/6 views and 0/3 pairs. Using all 174
labels, top-one predictions match the annotated direction labels on 2/6 and
0/6 views, respectively. These observations describe one recording.
[Pinned V-JEPA 2 checkpoint](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2/tree/4aa02df83918538fc21cfaf576382fa20e489a80).

Video *Collections in Motion: Reversible Fan*, Cooper Hewitt,
Smithsonian Design Museum, CC BY 3.0. The figure samples frames, outlines
the center crop, and shows the frames in both orders.
[Video source](https://commons.wikimedia.org/wiki/File:Collections_in_Motion_Reversible_Fan.webm) ·
[License](https://creativecommons.org/licenses/by/3.0/).
