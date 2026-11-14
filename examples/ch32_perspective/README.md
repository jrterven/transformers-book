# Chapter 32. What Changed, and What Remains?

Run the fixed experiments from the repository root with Python.

```sh
python -m examples.ch32_perspective.run_examples --output output/ch32.json
python -m unittest discover -s tests -p test_perspective.py -v
```

`make_history` builds a sequence that starts with one integer code and
then assigns zero or more nonempty route labels. The code stays fixed
throughout the history. `compare_memory` presents that history to four
stores.

| Store | What it retains | Unit counted |
| --- | --- | --- |
| Full archive | Every event | Event |
| Recent window | Up to `window` most recent events | Event |
| Route-only summary | The latest route, if one has been assigned | Field |
| Code and route summary | The initial code and the latest route, if assigned | Field |

Each answer is read only from its store. Missing information gives
`None`. The counts describe how many events or fields are retained,
and those units can occupy different numbers of bytes.

The worked histories begin with codes 731 and 954 and then share
route assignments A, B, and C. At window width three, both the recent
window and the route-only summary return `C` for the route and `None`
for the code. The sweep compares one, two, four, and eight route updates
for the two initial codes. It scores the code and route answers
separately by exact match. The stores use Python's standard library and
fixed rules for reading and updating their contents.

`causal_pair_counts` counts full and local query–key pairs in a prefix,
including each query's self-pair. For a window of three it returns
`(36, 21)` at length eight and `(136, 45)` at length sixteen. These are
connectivity counts.

## Read the units before comparing

An event and a summary field are different logical units. Python integers may
grow and strings may have different lengths; constant field count alone does
not prove constant byte storage. An unavailable answer uses `None`, while
zero is a valid code. Inspect the input and rule to distinguish absent,
omitted and evicted information.

A width-three window retains the first code through two route assignments;
the third assignment fills the window with routes. A two-field summary answers
the two chosen questions but cannot reconstruct all intermediate routes.
The sweep uses two specified histories per length, not a learned model.

See the [12 exercises](EXERCISES.md) and
[complete solutions](../../solutions/ch32_perspective/README.md).

## Measure history and test evidence in a pretrained hybrid

The [recorded run](records/qwen_history.json) uses Qwen3.5-0.8B-Base
on CPU in float32. The runner counts the attention KV tensors, the
Gated DeltaNet states, and the convolution buffers after prefixes
of 32, 64, 128, 256, and 512 tokens. The last two components total
19.6875 MiB at each measured length. The attention payload grows by
24 KiB per token.

The runner constructs 16 parcel logs, crossing four pairs of
code names, two register assignments, and both final selections.
It restricts the next-token choice to two verified single-token
completions. The record gives both raw logits, probabilities normalized
over the two candidates, and their combined mass over the full
vocabulary. A pair passes when both choices are correct.

| Evidence supplied | Individual choices | Assignment swaps | Selection flips |
| --- | ---: | ---: | ---: |
| Full log | 8/16 | 0/8 | 0/8 |
| Recent log | 8/16 | 0/8 | 0/8 |
| Moved register | 7/16 | 0/8 | 0/8 |
| Selected entry | 16/16 | 8/8 | 8/8 |

In the last condition, a deterministic lookup supplies the entry for
the final selection. The test measures the model's choice given that
entry. These counts describe the 16 logs and two candidate tokens used
in this experiment.

Rebuild both figure PDFs from the record with Python 3.12 and
Matplotlib 3.11.2.

```sh
python -m pip install matplotlib==3.11.2
python -m examples.ch32_perspective.make_figures --output-dir output/ch32-figures
```

To repeat inference, use the CPU build of PyTorch for your platform and
the package versions in the record. The runner downloads roughly
1.6 GB of weights from the pinned revision and checks the weights,
config, and tokenizer against their saved hashes. An existing
snapshot can be supplied with `--checkpoint`.
[Pinned model revision](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68)

```sh
python -m pip install torch==2.15.0 transformers==5.21.0 huggingface-hub==2.4.0 \
  numpy==2.5.4 safetensors==0.8.0
python -m examples.ch32_perspective.run_pretrained --output output/ch32-qwen.json
```
