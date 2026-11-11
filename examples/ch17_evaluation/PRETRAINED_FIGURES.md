# Measured evaluations for Chapter 17

Redraw the five figures from the saved runs in the companion repository.

```sh
python -m pip install -r requirements-ch16-figures.txt
python -m examples.ch17_evaluation.pretrained_figures --output-dir output/ch17-real
```

The program writes the PNG figures and `evaluation.json` with the
per-message calculations, the resampling distributions, and the context
counts. The two inputs are the [Chapter 16 run record](../ch16_adaptation/PRETRAINED_FIGURES_RUN.json)
and [the context run](CONTEXT_RUN.json). Each input's SHA-256 is recorded in
the output.

## Evaluate saved choices and text

The base, SFT, and DPO checkpoints see the same 32 fictional requests. For
each request, the model scores both complete label strings with EOS, then
generates up to eight tokens using the original vocabulary. Candidate
probabilities condition on the two complete strings. Exact label accuracy
for the generated text counts each invalid string as an unsuccessful case.

The paired comparison uses the base and SFT candidate choices. All four
accounts have the same correctness difference within each wording family.
The exact distribution of the sampled mean follows by repeatedly convolving
the mass on differences -1, 0, and +1. Resampling messages or whole families
describes different reweightings of these observed cases, with the four
accounts fixed. The family range describes sensitivity to wording in this
fixture.

## Run the context probe again

```sh
python -m examples.ch17_evaluation.context_evaluation \
  --cache-dir output/model-cache --output output/ch17-context.json
python -m examples.ch17_evaluation.pretrained_figures \
  --context-run output/ch17-context.json --output-dir output/ch17-real
```

The first command uses the published `HuggingFaceTB/SmolLM2-135M` checkpoint
at revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Use `--offline` if
that revision is already cached. The run uses float32 inference on a CPU
with four threads and PyTorch's SDPA backend. The included record contains
the checkpoint digest, runtime versions, and native next-token logits.

Eight fictional report families have 16, 64, or 256 records. In each family,
insert the same queried account at the early, middle, or late position,
and run once for each code. A fourth placement fixes the eight distractor
records after the evidence across lengths. That suffix is 72 tokens. Each
64-record report also has a control that replaces the target record with
an unrelated account of the same token length. This makes 192 prompts with
evidence and eight controls. The two answer tokens encode ` red` and ` blue`.
The record contains the native top token, its text, the two candidate logits,
and their combined probability in the full vocabulary.

The fixture verifies each fully formatted prompt's token boundaries and
total length before inference. The record stores zero-based token indices
and a digest of the token IDs as little-endian int32 values. Each pair holds
the question and distractors fixed while changing one code. These results
measure the supplied register task with this model, prompt, and answer rule.
