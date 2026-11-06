# Chapter 21. Image and Video Understanding

Run the offline examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch21_vision.run_examples --output output/ch21.json
python -m unittest discover -s tests -p test_vision.py -v
```

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
