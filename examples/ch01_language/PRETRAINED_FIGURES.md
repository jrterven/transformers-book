# Observe a pretrained language model

The chapter's three additional figures inspect the actual tokenizer, embedding
table, final hidden states, and vocabulary logits of
[`distilbert/distilgpt2`](https://huggingface.co/distilbert/distilgpt2), at revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`. The model card labels the checkpoint
Apache 2.0. The experiment uses built-in Transformers implementations and
safetensors weights, without loading remote Python. No training or sampling
is performed. The regular NumPy example needs none of these optional packages.

From the companion repository root:

```sh
python -m pip install -r requirements-appendix-figures.txt
python -m examples.ch01_language.pretrained_figures \
  --cache-dir /path/to/huggingface/hub \
  --output-dir output/ch01-pretrained
python -m unittest discover -s tests -p test_ch01_pretrained_figures.py -v
```

The first run downloads the pinned checkpoint to the specified cache directory.
Add `--offline` once those files are cached. The
[committed run record](PRETRAINED_FIGURES_RUN.json) contains the exact inputs,
token IDs, results, package versions, weights SHA-256, and source/output hashes.
Different platforms or package versions can change floating-point results.
The script checks the scientific properties rather than requiring bitwise
agreement with one machine's output.

The three observations answer different questions:

1. **What text does a token represent?** Tokenize `the red car`, `The red car.`,
   `unhelpfulness`, `red`, and ` red`, without adding special tokens. Decode
   the complete ID sequences to verify exact round trips. A middle dot in
   the figure marks an actual space in a decoded piece. IDs belong to this
   checkpoint's 50,257-entry vocabulary, not the chapter's 16-entry toy table.
2. **What changes when the context changes?** Read the final ` car` token's
   lookup row and final hidden vector in `the red car` and `the blue car`.
   The token ID and position are the same. All 768 lookup coordinates match;
   the contextual vectors differ. The image displays the first 12 coordinates
   directly, without PCA or other projection. Each panel uses one shared signed
   color scale for its two rows, with separate scales between panels. No
   coordinate is assigned a semantic name. As a causal control, changing only
   the later noun in `the red car` to `bike` leaves the earlier hidden vector
   at ` red` unchanged within an absolute tolerance of `1e-6`.
3. **Can an earlier verb affect the next token?** Compare `She drove the red`
   and `She wore the red`. The last two tokenizer IDs match exactly. Inspect
   the single-token continuations ` car`, ` dress`, ` truck`, and ` shirt`.
   These four fixed candidates illustrate the question; they are not the
   four most likely entries or a benchmark. Normalize over all 50,257 logits
   and retain the sum for the other 50,253 entries. Both candidate charts use
   the same probability axis; the lower 100% bars expose the remaining mass.

All forward passes use frozen parameters in evaluation mode on a CPU with
float32 arithmetic. The recorded float32 logits are converted to float64 for
the full-vocabulary softmax. There are no token filters, temperature changes,
top-k operations, or subset renormalization. The output contains three PNGs,
`ch01_pretrained_observations.npz` with the full observed vectors/logits/
probabilities, and `PRETRAINED_FIGURES_RUN.json`. The vectors are observations
of internal computation; the probabilities grade continuations, not truth.

For background on spaces and byte-level BPE, see the
[versioned GPT-2 tokenizer documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/gpt2#transformers.GPT2Tokenizer).
The hand-worked BPE merges in the manuscript are a separate character-level
teaching example; they are not this checkpoint's learned merge rules.
