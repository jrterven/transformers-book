# Chapter 22. Image and Video Understanding

Run the offline examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch22_vision.run_examples --output output/ch22.json
python -m unittest discover -s tests -p test_vision.py -v
```

The [measured visual demonstrations](PRETRAINED_FIGURES.md) follow a patch
through DINOv3 and compare V-JEPA 2 action scores for original and reversed
clips. The guide includes media credits, pinned checkpoints, and the commands
to regenerate the four figures.

## Turn pixels into patch tokens

`patchify` expects one channels-last image `[height, width, channels]`.
Both spatial dimensions must be divisible by the square patch size.
Patch rows follow the image grid from left to right, then top to
bottom. Inside each row the pixel values are ordered by row, column,
and channel.

`patch_embeddings` applies the same projection to every flattened
patch. It can prepend a CLS vector and add a position vector to
every row, including CLS when present. The example's 8-by-8 RGB
image produces four rows of width 48 and then a 5-by-6 sequence.

## Probe fixed spatial features

The images come in pairs. Each pair uses the same background and
one bright patch, with the patch moved between upper and lower
positions in the same column. The two images therefore have the
same pixel histogram and different labels. Held-out pairs use
different brightness and background settings.

The feature extractor computes one mean per patch. An ordered row
keeps the four means in patch-grid order. Sorting the same four
means keeps their values and removes their locations. Both heads
use the same feature width, zero initialization, and optimizer.
`train_linear_probe` updates only the affine head, while
`probe_metrics` evaluates its accuracy and cross-entropy.

## Count attention pairs in a video

`space_time_pairs` counts ordered query-key pairs with self-pairs
included and special tokens excluded. Joint attention pairs all
patches across all frames. Divided attention adds a spatial count
within each frame and a temporal count at each patch location.
For three frames with four patches each, the joint count is 144
and the two divided counts sum to 84.

## Understand what the representation preserves

Flattening rearranges pixels; it does not compress them. The learned
projection can. A list's row order alone does not give a position-free
full-attention encoder access to location. Fixed position vectors associate
patch content with its grid location. The ordered mean features used here
are hand-designed spatial coordinates, not a pretrained ViT's output.

The paired sorted features impose a 50% ceiling on deterministic binary
classification: each identical feature row has opposite labels. This is
an information limit, not simply an unsuccessful optimizer. The supplied
runner obtains 8/8 versus 4/8 on eight held-out synthetic images.

In temporal-then-spatial attention, information can travel from one frame
and location to a different frame at the same location, then to a different
location in that frame. The intermediate representation is the output of
the temporal stage. Patch-only divided/joint score counts have ratio
`1/T + 1/S`; counts do not predict runtime or imply identical attention maps.

See the [12 exercises](EXERCISES.md) and [complete solutions](../../solutions/ch22_vision/README.md).
All editions include every solution, with plots for patch order and video
counts. Evaluation designs cover recording and camera leakage, frame
reversal with temporal position information, and controlled resolution changes.
