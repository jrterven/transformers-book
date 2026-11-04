# Chapter 5: assembling the Transformer

Juan Terven and Diana Córdova

This PyTorch implementation builds a small encoder-decoder Transformer from
visible components: sinusoidal positions, multi-head attention, positionwise
feed-forward networks, residual paths, post-norm layers, and a vocabulary head.
It follows the block ordering of the 2017 architecture. The worked model is
randomly initialized and runs on a CPU; it is not a trained translation system.

## Run the examples

From the companion repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-ch05.txt
python -m examples.ch05_transformer.run_examples
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
```

Chapter 4 needs only NumPy. The Chapter 5 test class is skipped if PyTorch is not
installed; installing `requirements-ch05.txt` enables all component and model
checks. No pretrained model, dataset, accelerator, or training loop is needed.

## What the runner calculates

1. A deterministic two-head example reuses Chapter 4's input matrix. The query
   and key projections are identity matrices, and the value projection swaps
   the two feature halves. Head 0 (the first head) reproduces Chapter 4's attention.
   The runner prints both attention matrices, the concatenated outputs, and the
   result of the book's explicit output projection.
2. Sinusoidal encodings are printed for positions 0, 1, and 2 at width four.
3. A hand-worked feed-forward example uses row-vector matrices `W1=[I,-I]` and
   `W2=[I;I]`, with zero biases. Its hidden ReLU output is
   `(1,0,0,3,0,2,0,0)` and its output is `(1,2,0,3)`.
4. A complete model uses width 4, two heads, FFN width 8, one encoder layer, and
   one decoder layer. With seed 7, it computes logits and one illustrative
   cross-entropy, calls `backward()`, and checks finite gradients. This check
   verifies connectivity; it does not establish that the model has learned.

For row-vector multiplication `x @ W`, `nn.Linear` stores `W.T`. The attention
implementation uses batch-first inputs and explicitly splits and merges heads.

## API and shapes

`model.py` provides:

| Component | Interface |
| --- | --- |
| `PositionalEncoding(d_model, max_length)` | Adds fixed positions to `[B,N,d_model]`; embedding scaling is done by the model. |
| `MultiHeadAttention(d_model, n_heads)` | `(query,key,value,allowed=None) -> (output,weights)`. |
| `PositionwiseFeedForward(d_model,d_ff)` | Applies the same two linear maps and ReLU at every position. |
| `EncoderLayer` | Self-attention and FFN, each followed by residual addition and layer normalization. |
| `DecoderLayer` | Causal self-attention, cross-attention, and FFN, each with post-norm wrapping. |
| `build_masks(source_valid,target_valid)` | Returns source-key and causal-target-key allowed masks. |
| `Transformer(source_vocab_size,target_vocab_size,...)` | Returns vocabulary logits; `return_weights=True` also returns attention matrices from each layer. |

For source length `S`, decoder-input length `T`, batch size `B`, and `h` heads:

```text
source token ids:       [B, S]
decoder input ids:      [B, T]
encoder self-attention: [B, h, S, S]
decoder self-attention: [B, h, T, T]
cross-attention:        [B, h, T, S]
vocabulary logits:     [B, T, target_vocab_size]
```

Each head has width `d_model // h`; this implementation requires divisibility.
The attention module supports different query and key sequence lengths. Keys
and values must share their batch size and length, and all input feature
widths equal `d_model`. The returned weights are normalized probabilities;
no attention-probability dropout is applied.

## Masks and the shifted target

Masks are Boolean and **`True` means allowed**. This is the convention used by
our implementation and by PyTorch functional scaled dot-product attention;
other PyTorch attention APIs can use the opposite convention. Do not transfer
masks between interfaces without checking.

`source_valid` and `target_valid` are Boolean `[B,length]` tensors. They describe
nonempty right-padded prefixes; source examples and decoder inputs must contain
at least one valid token. The model defaults to `ids != pad_id`, or accepts
explicit validity tensors independently of the ids. This lets the tests change
padded token ids while preserving their masked status.

Source-key masks have shape `[B,1,1,S]` and are shared by encoder self-attention
and decoder cross-attention. Target masks combine the lower triangle, including
the diagonal, with target-key validity, giving `[B,1,T,T]`. A row that excludes
every key raises `ValueError`. Key masks do not zero padded query outputs;
invalid target labels must be excluded from the objective separately.

The demonstration uses separate source and target vocabularies of size 12.
Only the following ids matter to the example:

```text
both:    0=PAD, 1=BOS, 2=EOS
source:  3=the, 4=red, 5=car, 6=a
target:  8=el, 9=coche, 10=rojo, 11=un

source ids:  [[3,4,5,2,0], [6,5,2,0,0]]
input ids:   [[1,8,9,10],  [1,11,9,0]]
label ids:   [[8,9,10,2],  [11,9,2,0]]
```

For the first example, the decoder sees `BOS el coche rojo` and predicts
`el coche rojo EOS`. The last padded label in the second example is excluded by
`ignore_index=0`. `cross_entropy` receives logits directly; applying softmax
before it would be incorrect for this API. The runner does not call an optimizer.

## Scope and checks

The code uses post-norm blocks, ReLU FFNs, sinusoidal positions, and embedding
scaling by `sqrt(d_model)`. Layer normalization uses the population variance
across the last axis with epsilon `1e-5`. Input dropout follows the sum of scaled
embeddings and positions; sublayer-output dropout precedes residual addition.
`eval()` disables dropout while preserving autograd. `torch.no_grad()` is a
separate choice.

The small configuration does not reproduce the original training recipe. It
uses independent source and target embeddings and an untied output projection,
rather than the original shared embedding/output weights. It omits dropout on
attention probabilities and inside the FFN. Every stacked layer owns independent
parameters. There is no final pre-norm-style normalization, KV cache, optimizer,
label smoothing, or generation loop here.

Tests compare multi-head attention with PyTorch's `nn.MultiheadAttention`,
copying the learned projection weights and inverting the Boolean mask. They
verify head ordering and the hand-worked numbers, inspect
post-norm placement and population variance, and check padding and causal
information flow through the complete model. They also check independent layer
parameters, evaluation/training dropout behavior, shifted labels, and gradients.
An additional full-stack regression compares two encoder and two decoder layers
against PyTorch's reference layers, including outputs, input gradients, and all
62 corresponding stack and vocabulary-head parameter-gradient tensors.

The runner's `check_prefix_and_padding` helper is specifically for the batch
returned by `example_batch()`: padding ID 0, the illustrated token positions,
and the 12-entry vocabularies. It switches the model to evaluation mode and
supports different stack depths. For other batches or vocabularies, adapt its
perturbed positions, replacement IDs, and validity masks. The Transformer
class itself supports other vocabulary sizes and padding IDs.

The eight book listings are exported from the marked executable source sections:

```sh
python scripts/export_snippets.py --chapter ch05
python scripts/export_snippets.py --chapter ch05 --check
```

Method excerpts omit their class constructors and some validation helpers; run
the complete modules rather than individual snippet files. The exporter also
supports `--chapter ch04` and defaults to both chapters.
