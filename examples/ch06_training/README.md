# Training and Generating with a Small Transformer

This chapter trains the encoder-decoder from Chapter 5 to reverse short
sequences of colors. The task is deliberately small: it exposes shifted
targets, token-weighted loss, validation, autoregressive generation, and KV
caching without downloading a dataset or pretrained weights.

Run these commands from the repository root:

```sh
python -m pip install -r requirements-ch06.txt
python -m examples.ch06_training.run_experiment --output output/ch06
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
```

The experiment uses one CPU thread. Its output directory contains a selected
model checkpoint (`best.pt`) and the complete recorded configuration, validation
history, held-out metrics, example predictions, and timing results
(`metrics.json`). These generated outputs are ignored by Git.

A one-update run checks the command path before the longer experiment:

```sh
python -m examples.ch06_training.run_experiment --steps 1 --output output/ch06-smoke
```

Use a separate directory for this smoke run. It checks data construction,
checkpoint selection, generation, and timing, rather than learned accuracy.

## Data and target construction

The vocabulary is fixed before splitting:

| ID | Token |
| --- | --- |
| 0 | `<pad>` |
| 1 | `<bos>` |
| 2 | `<eos>` |
| 3–8 | `red`, `green`, `blue`, `yellow`, `white`, `black` |

`make_splits(seed=2026)` returns 512 training, 64 validation, and 64 test
sequences. Every split has equal counts of lengths 3, 4, 5, and 6. Repeated
colors are allowed. Source sequences are unique and disjoint across splits;
the target is the exact reversed source. The seed and split SHA-256 hashes
are saved with the checkpoint so the data can be reproduced.
The generator samples sequence indices rather than constructing every possible
sequence. Trying longer sequences therefore needs memory for the sampled
examples, not for all six-to-the-length combinations.

For the source `red green blue`, `collate` creates:

```text
source:         red   green blue
decoder_input:  <bos> blue  green red
target:         blue  green red   <eos>
```

The three arrays are right-padded independently. Source padding and target
input padding determine attention masks. Target padding determines which
predictions enter the loss. The diagonal of causal self-attention remains
allowed: at each position, the input token precedes its aligned label.

## Training and evaluation

The recorded recipe uses model width 64, four heads, feed-forward width 128,
two encoder layers, two decoder layers, and dropout 0.0. All matrices receive
Xavier-uniform initialization; the LayerNorm and vector-parameter defaults
remain as defined by Chapter 5. Adam uses learning rate 0.001 and betas
`(0.9, 0.999)` for 2000 updates of 64 randomly sampled training sequences.
The model seed and data seed are both 2026. Validation is checked every 100
updates. This small teaching recipe differs from the original 2017 training
schedule, label smoothing, dropout, and model scale.

`token_loss(logits, targets)` passes raw logits to PyTorch cross-entropy and
returns a sum of valid-token losses plus their count. Padding **positions**
are ignored; the padding vocabulary class remains in each valid softmax
denominator. The end token counts as a target; the start input does not.

We can inspect those rules without training anything:

```python
from examples.ch06_training.run_experiment import loss_sanity_example

print(loss_sanity_example())
```

This separate arithmetic batch contains sources `red green blue` and `red blue`.
Their target rows have four and three real labels after adding EOS, plus one
padded storage position. Uniform logits have shape `[2,4,9]`. Every evaluated
label receives probability `1/9`, so the loss sum is `7 * log(9)`, the count is
7, the token mean is approximately `2.197225`, and perplexity is 9. The
two-color source illustrates padding; the actual training split still uses
lengths three through six. The CLI prints this check before training.

`train_step` enables training mode, clears gradients, takes the token mean,
backpropagates, clips the gradient norm to 1.0, and updates Adam. Validation
runs in evaluation mode without gradients. Corpus NLL is the total loss
divided by the total token count, rather than an average of batch means.
Perplexity is `exp(NLL)` with natural logarithms. Token accuracy uses supplied
reference prefixes; exact-match accuracy uses greedy generated prefixes and
requires the correct complete sequence **including EOS**. A length-capped
output without EOS is not an exact match.
Termination rate separately counts the fraction of generated rows containing
EOS. The final metrics record also breaks held-out results down by source
length, making repetition, premature stopping, and length-dependent failures
easier to investigate.

`evaluate(model, sequences, batch_size=64, generation_max_new_tokens=None)`
uses one output-length limit for the entire evaluation dataset. Changing the
minibatch size therefore does not give some examples more chances to emit EOS.
The default limit is the longest source length plus two (EOS and one extra
token), capped by the model's positional table. Pass an explicit
`generation_max_new_tokens` to compare different datasets under the same limit.

The selected checkpoint minimizes validation NLL at the recorded evaluation
steps. Test data are evaluated only after selection. The checkpoint contains
model weights, architecture, vocabulary, experiment configuration, and selected
validation metrics. `load_checkpoint` uses `weights_only=True` and reconstructs
the model on CPU while preserving the saved parameter dtype. It is an inference
checkpoint; it does not contain optimizer state for resuming training.

Load the selected model and try a source sequence:

```python
from examples.ch06_training.data import collate, words
from examples.ch06_training.decoding import greedy_decode
from examples.ch06_training.training import load_checkpoint

model, metadata = load_checkpoint("output/ch06/best.pt")
batch = collate([(3, 4, 5)])  # red green blue
prediction = greedy_decode(model, batch.source)
print(words(prediction[0], include_special=True))
# blue green red <eos> for the recorded checkpoint
```

## Generation and cache conventions

Call `model.eval()` before generation. `greedy_decode` also disables gradients,
encodes the source once, and begins with BOS. It suppresses PAD and BOS as
generated tokens, stops each row at its first EOS, and pads finished rows in
the returned output while other rows continue. A shared maximum output length
provides a termination bound.

The uncached path re-evaluates the complete target prefix while reusing encoded
source memory. The cached path reuses the same model weights and stores each
decoder layer's projected self-attention keys and values. It projects the
current token once, appends its K/V before attention, and uses its absolute
sinusoidal position. Each layer also stores its own fixed cross-attention K/V
projections of the encoded source. Source padding remains masked.

`CachedDecoder.step(token)` accepts a single non-PAD token column `[B, 1]`
and returns logits `[B, V]`. Caches are per request and must be reset for a
different source or prefix. Cached self-KV tensors have shape
`[B, H, prefix_length, head_width]`; cross-KV has source length in that axis.
The reference uses `torch.cat` to grow tensors, so it exposes the mechanism
without attempting production memory allocation or kernel optimization.

When comparing a padded batch of reference prefixes with the cache, do not
pass PAD to `step`. The runner replaces the shorter rows' trailing storage
with repeated EOS for that comparison, and applies the same replacement in
the full-prefix reference. For real generation, finished rows already feed
repeated EOS internally while the returned output pads their completed rows.
Compare logits with a numerical tolerance: equivalent matrix calculations can
round differently between one-token and complete-prefix execution.

The chapter also asks what goes wrong if target self-attention is allowed to
read future reference inputs. Run that experiment in a copy of the project
and start a fresh Python process after changing the shared `build_masks`
function in Chapter 5's module. Both the model and `decode_prefix` must use
the changed function; changing only one already-imported name does not update
the other module's reference.

Measure the broken model's teacher-forced loss directly with its logits and
`token_loss`, and use `greedy_decode(..., cached=False)` for free generation.
The ordinary `evaluate` helper selects cached generation, which is valid for
the original causal model. With noncausal attention, extending a prefix can
change earlier states, so its old keys and values cannot simply be reused.
Restore the causal rule before using the normal evaluator or cache benchmark.

## Timing

`benchmark` reports median CPU milliseconds for source encoding, cached target
prefill, one full-prefix next-token calculation, and one cached next-token
calculation. It uses a fixed supplied prefix, warms the model, and reconstructs
the cache outside each decode timing. The reported target prefill consumes all
but the last supplied token through successive one-token cache steps; it is
not a parallel prefill kernel. Sampling, data loading, and text conversion are
outside the timed region. Source encoding and target prefill are setup costs,
so the per-token cached timing is not an end-to-end latency number. Tiny
single-thread CPU measurements are noisy and do not predict GPU speedups.

The complete local validation record is in [VALIDATION.md](../../VALIDATION.md).

The nine book listings are exported from the canonical source files in this
directory. Run `python scripts/export_snippets.py --chapter ch06` to regenerate
them, or add `--check` to check their correspondence without writing.
