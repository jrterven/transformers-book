# Observations of pretrained models

These figures connect the background appendices to actual learned weights.
They use public checkpoints with pinned revisions or file hashes. They do not
train the networks or measure generation quality or execution speed.

- **A:** two attention heads in DistilGPT2, including the causal zeros and full
  row normalization.
- **B:** its observed next-token probabilities and the derivatives of a supplied
  target's cross-entropy. The probabilities use the entire vocabulary; the plot
  shows a subset. The backward calculation differentiates the final loss only,
  and checks every derivative against `p - y`.
- **C:** an FP32 common-offset check on the same full logit row. Stable softmax
  remains finite; input rounding from the offset is measured separately.
- **D:** logical K/V tensor payloads at two supplied lengths, checked against the
  shape formula. This excludes backing-storage capacity, weights, temporary
  arrays, and allocator reservations.
- **E:** the first convolution of ResNet-18 using ImageNet-1K V1 weights. Its input
  is the original cat image generated for Appendix F, not a real photograph.
  No full-image classification is performed. Responses are captured before
  batch normalization and ReLU.

Install the optional dependencies from the companion repository root:

```bash
python -m pip install -r requirements-appendix-figures.txt
python -m examples.appendix_models.inspect_models --output-dir output/appendix-models
```

The first run downloads DistilGPT2's safetensors and tokenizer files from
[its public repository](https://huggingface.co/distilbert/distilgpt2), at revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`. Its model card labels the weights
Apache 2.0. No remote Python is loaded. ResNet's weights come from the
[official Torchvision weights specification](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
and are checked against the full SHA-256 in the script. Only its first-layer
weights are evaluated, using PyTorch's cross-correlation convention.

Pass `--cache-dir /path/to/cache` to choose where to store the weights. Once
downloaded, add `--offline` to rerun without network access. Inference and all
reported observations use CPU float32. The loss derivatives are evaluated in
float64 from the recorded float32 logits. Evaluation mode disables dropout;
the model weights remain fixed. The supplied target ` mat` is an illustrative
training label, not a unique correct answer to the prompt.

The output includes three PNGs, `observations.npz` with unscaled arrays, and
`manifest.json` with the input IDs, model revision, selected heads/channels,
shapes, results, software versions, display settings, and source/output hashes.
The committed [run record](RUN.json) describes the book's run. Different software
versions can change floating-point results; test properties and stated
tolerances rather than assuming bitwise agreement across platforms.

Attention plots use a common scale of 0 to 1. Convolution panels use a common
signed range determined by the largest absolute displayed response. There is
no independent contrast normalization of each panel. Channels 1, 14, and 61 are
selected for illustration; they are not a statistical sample of model features.
The PNG input is a fixed 8-bit rendering of the Appendix F seed-20261110 output,
using `round(255 * clip(x / 2 + 0.5, 0, 1))`. The image is center-cropped to
224 by 224, rescaled to 0--1, and normalized with the model's specified means
and standard deviations. Source-image provenance is also recorded in the
[diffusion run manifest](../ch25_visual_generation/DIFFUSION_FIGURES_RUN.json).
