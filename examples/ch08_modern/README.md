# From the Original Block to Modern Decoders

Chapter 8 builds a small causal decoder from pre-RMSNorm residual updates,
SwiGLU, and rotary self-attention. It uses the token vocabulary and shifted
labels from Chapter 7. The entire example runs on a CPU with the PyTorch
requirements used in Chapter 5.

The optional [pretrained-model inspection](PRETRAINED_FIGURES.md) adds four
measured figures from a pinned SmolLM2 checkpoint. They trace actual RMSNorm,
SwiGLU, rotary scores and masked attention, and residual updates across a
30-block model. Independent calculations check native outputs. Its optional
dependencies are in `requirements-ch08-figures.txt`; the teaching decoder
below remains independent of pretrained weights.

Run these commands from the repository root.

```sh
python -m pip install -r requirements-ch05.txt
python -m examples.ch08_modern.run_examples
python -m examples.ch08_modern.run_examples --output output/ch08.json
python -m unittest discover -s tests -p 'test_modern.py' -v
```

The program prints the worked examples, then runs a forward pass and one
optimizer step with a fixed seed. The optional output path saves the same
JSON record.

Work through the [14 exercises](EXERCISES.md) in order: hand calculations,
small model probes, and three optional investigations. The
[complete solutions](../../solutions/ch08_modern/README.md) are available as
an executed notebook, a written PDF, and a PDF including the code and outputs.

## Follow the component examples

| Function | Calculation | Result |
| --- | --- | --- |
| `normalization_example()` | Normalize `(3, 4)` with unit gains and epsilon zero | LayerNorm `(-1, 1)`, RMSNorm about `(0.849, 1.131)` |
| `swiglu_example()` | Activate the gate `(0, 2)` and multiply by `(3, -1)` | Product about `(0, -1.762)`, projected output about `(0, -1.762, 0, 1.762)` |
| `rotation_example()` | Rotate two unit vectors with a 30-degree step and shift both positions by two | Both pairwise dot products are 0.5 to numerical precision |
| `pair_score_example()` | Sum two pairwise contributions at frequencies 1 and 0.01, then divide by the square root of head width 4 | Contributions about -0.416147 and 0.999800; scaled logit 0.291827 |

`rms_norm` averages squares across the last axis, so each token gets its own
scale. It computes statistics in float32 for float16 and bfloat16 inputs and
keeps float64 inputs in float64. The module supplies a learned gain per
feature and uses epsilon $10^{-6}$ by default. The hand calculation uses
epsilon zero on the nonzero vector.

With zero epsilon and unit gains, a nonzero width-$d$ vector has output
length $\sqrt{d}$, while its mean square is one. RMSNorm preserves direction
before the learned gains and does not remove a common additive shift.

`SwiGLU` sends the same token through independent gate and up projections.
It applies SiLU to the gate, multiplies the two branch outputs coordinate by
coordinate, and uses the down projection to return to the model width.
PyTorch stores a linear weight as `[out_features, in_features]`. Its weight
array is the transpose of the matrix written to the right of a row vector
in the chapter.

SiLU gates can be negative or greater than one; they are feature multipliers,
not probabilities. The same projection weights are used at every position,
but each token's current features determine its multipliers.

`rotate_pairs(x, positions)` takes `x` shaped
`[batch, heads, tokens, head_width]` and one integer position per token. The
indices are shared across the batch and heads. The function rotates adjacent
feature pairs using frequencies $10000^{-2r/d_h}$ for pair $r$ and head
width $d_h$. The rotation illustration above chooses 30 degrees per
position for easy geometry. The model uses the frequency formula. Attention
rotates queries and keys, then combines the unrotated values using scaled
dot-product weights.

## Assemble and check the decoder

`ModernBlock` adds the attention update to its original input, then adds
the feed-forward update to that result. Each branch reads its own normalized
input. `ModernDecoder` embeds IDs, applies two independent blocks by default,
and sends the final normalized vectors to the vocabulary projection.

The input IDs have shape `[batch, tokens]`. The Chapter 5 mask builder
restricts keys to earlier or current positions and excludes padding keys.
Every row must contain a nonempty prefix followed by any padding. Padded
query positions still produce outputs. Their label slots use `IGNORE=-100`
so the loss grades only the five selected positions in the demo batch.

The example has width 16, two heads of width 8, intermediate width 32,
two blocks, and a vocabulary of 17 tokens. Each block has 2,592 parameters
from four attention projections, three feed-forward projections, and two
normalization gains. The embedding table, separate vocabulary projection,
and final gain bring the total to 5,744.

The recorded one-thread float64 run gives logits of shape `[2, 3, 17]`.
Changing a later input leaves the earlier logits equal. Appending padding
or shifting the position indices of the whole sequence changes real logits
by less than $10^{-12}$ in the example. The tests also inspect attention
weights and check that padding keys receive zero weight. `start_pos` sets
the starting index for the supplied sequence. The example recomputes the
whole supplied prefix on each call.

Appending right-padding alone does not isolate the padding-key rule: the new
keys are also in the future of all original queries. Exercise 9 inspects the
last diagonal mask entry, which a causal-only mask would incorrectly allow.
Exercise 11 distinguishes relabeling all positions from prepending an actual
token: the latter adds a key and value and can change the logits. This symmetry
check is not an evaluation of long-context prediction quality.

The teaching width 32 matches a two-projection feed-forward width of 48 at
the same parameter count. It is not the rounded $8d/3$ choice used for matching
a conventional width of $4d$. RoPE adds no learned position parameters here,
and no absolute position embeddings are added to the token embeddings.

`one_update` uses the Chapter 7 loss helper and one SGD step with learning
rate 0.01. Every parameter must receive a finite gradient. A missing or
nonfinite gradient raises `FloatingPointError` before the optimizer can
change the weights. In the recorded run, the mean loss on the fixed batch changes
from 3.245774 to 3.197668, and every parameter gradient is finite. Use the
fixed batch to trace which tokens contribute to the loss and how gradients
reach the projections and gains.
