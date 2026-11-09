# Chapter 11. Long Context, Sparse Access, and Sequence Compression

Run from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch11_long_context.run_examples --output output/ch11.json
python -m unittest discover -s tests -p test_long_context.py -v
```

All positions in the code and JSON are zero-based.

The optional [pretrained inspection](PRETRAINED_FIGURES.md) runs a pinned
Longformer checkpoint on a 648-token report. Four measured figures show
local and global probability maps, actual relay reads, the propagation of
a one-token change, and retained-mass versus head-output error. This model
download is independent of the regular NumPy examples below.

## Make a pattern of allowed connections

`local_mask` places queries on rows and keys on columns. It first marks
the keys within the given radius. A global position can read or be read
from anywhere in the bidirectional pattern. With `causal=True`, every
connection must also respect the ordering of the tokens. The returned
Boolean grid is a way to inspect connectivity.

## Choose entries, then compute their weights

`selected_attention` takes one projected query, matrices of key and
value rows, and a separate vector of index scores. The optional Boolean
`allowed` vector limits the candidates before ranking. The function
takes up to `k` positions, gathers their key and value rows, and calls
the scaled dot-product attention function from Chapter 4. When index
scores tie, the earlier candidate ranks first. The returned weights
are in the same order as the selected positions. Ranking uses the
input score dtype so that integer scores retain their exact order.

The worked example selects indices 1 and 6. Their main attention
weights are 0.75 and 0.25. The causal check gives index 7 a larger
ranking score and excludes it with the mask. Selecting every allowed
entry gives the same output as the masked dense reference up to
floating-point rounding.

## Summarize blocks and combine them with raw entries

`block_summaries` receives two arrays of shape `[time, width]`, one for
the projected features and one for their gate logits. Within each
window, a softmax over positions assigns weights independently to
each coordinate. The second return value records each window's last
position. An unfinished final block contributes no summary. In overlap
mode, each window contains its stride block and the preceding stride
block if one exists. Both the attention and summary examples expect
real numeric inputs.

The mixed-history example checks which summary windows are finished at
the current position, selects one of those summaries, and joins it with
two recent raw entries. Main attention uses one softmax over the three
gathered rows. These fixtures use hand-chosen scores, features, and small
arrays so the intermediate steps can be inspected.

## Work through the exercises

The [twelve exercises](EXERCISES.md) and [complete solutions](../../solutions/ch11_long_context/README.md) cover hand calculations, code, and experimental design. All exercises appear in the notebook and both PDFs. The notebook includes a causal relay counterexample, a joint-softmax comparison (17/3 versus 4.75), and a visual sweep of evidence selection versus output error.

The summary end-position rule assumes causal features and gates. It permits a completed summary at its last token; the DeepSeek-V4 report waits until the next block. The toy also omits learned normalization, RoPE, and the learned sink. Its purpose is to make visibility, pooling, and joint normalization inspectable.
