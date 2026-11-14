# Chapter 20 model demonstrations

Run these commands from the repository root. The figure command uses the
saved record and the manual fact audit. It needs NumPy and Matplotlib.

```sh
python -m pip install 'numpy>=1.26' matplotlib==3.11.2
python -m examples.ch20_generation.pretrained_figures
```

To run the checkpoints, use Python 3.11 or newer and the optional
requirements. The first run fetches about 1.94 GB of weights and the
tokenizer files. Subsequent runs can add `--offline` to use the cache.

```sh
python -m pip install -r requirements-ch20-figures.txt
python -m examples.ch20_generation.pretrained_observations --output output/ch20-pretrained.json
python -m examples.ch20_generation.pretrained_figures --record output/ch20-pretrained.json
```

The script pins the two model revisions and verifies the weight checksums.
It reads the official Marian tensor checkpoint with PyTorch's restricted
`weights_only` loader and the BART checkpoint from Safetensors. Both use
Transformers' built-in model classes. The recorded run used float32 on
four CPU threads with deterministic algorithms. The outputs were identical
in a second run on the same environment.
[OPUS-MT weights](https://huggingface.co/Helsinki-NLP/opus-mt-en-es/tree/5bc4493d463cf000c1f0b50f8d56886a392ed4ab)
and [BART weights](https://huggingface.co/facebook/bart-large-cnn/tree/37f520fa929c961707657b28798b30c003dd100b).

## What the runs record

- The same eight authored English sources go through greedy decoding and
  width-four search with raw and mean log-probability ranking. The runner
  records the Spanish text, token IDs, selected log-probabilities, lengths,
  completion flags, and one-reference SacreBLEU counts and signature.
- The blue-bus trace records last-block cross-attention averaged over its
  heads and scores for the supplied blue and red Spanish targets. Attention
  weights sum across source positions. Next-token probabilities sum across
  target vocabulary IDs.
- BART reads either all seven sentences of one authored report or its first
  four. Each input is decoded with limits of 18 and 72 generated tokens.
  The longer full-input output gives the fog as the cause of the delay and
  omits the boarding and cancellation facts. Both runs at limit 18 stop
  mid-sentence.
- ROUGE-L uses Unicode case folding and runs of letters or digits as tokens,
  with one reference for the full report. The manual audit records which
  required facts each output conveys and the source sentence for each fact.
  The figure renderer checks the audited output strings against the record.

[Recorded model outputs](PRETRAINED_RUN.json) and [fact audit](SUMMARY_FACT_AUDIT.json).

These inputs illustrate particular behaviors under fixed settings. Use a
separate representative evaluation to estimate performance on new reports.
