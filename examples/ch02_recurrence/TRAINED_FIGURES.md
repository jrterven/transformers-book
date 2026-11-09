# Look inside the trained recurrent model

These three manuscript figures observe the chapter's actual trained
42,857-entry LSTM encoder-decoder. It reverses lists of six familiar colors;
it has no attention and does not translate natural language. Its parameters
are learned from the synthetic training pairs, then frozen for inspection.
No outside model or dataset is downloaded.

From the companion repository root:

```sh
python -m pip install -r requirements-ch02-figures.txt
python -m examples.ch02_recurrence.run_examples --train --steps 750 --output output/ch02
python -m examples.ch02_recurrence.inspect_trained \
  --checkpoint output/ch02/best.pt --output-dir output/ch02-observations
python -m unittest discover -s tests -p 'test_recurren*.py' -v
```

The chapter uses seed 2026, one CPU thread, float32 parameters, and the
validation-selected checkpoint at update 600. The
[run record](TRAINED_FIGURES_RUN.json) records software versions, configuration,
checkpoint and ordered-parameter hashes, source hashes, and observed results.
Use that environment for the reference run. Other versions or devices can
produce different floating-point values and may select a different checkpoint;
the numerical and causal checks matter more than matching rounded image labels.
The inspection script explicitly checks that the selected example still
exhibits the reference failure rather than silently substituting another case.

The inputs and displays are fixed in the source:

- **State history:** `red green blue` and `yellow green blue`, each read from
  fresh zero hidden/cell states with the same encoder weights. Both sequences
  happen to be outside all three splits. They are illustrative probes, not
  extra scored test examples. Each heatmap shows coordinates 1–8 of 64 directly,
  with no projection or coordinate-wise normalization. Both hidden panels
  share a symmetric signed scale; both cell panels share a separate scale.
- **Actual gate arithmetic:** encoder coordinate 1, chosen before inspection,
  while reading `green blue black red red red`. All three gate multipliers are
  plotted. The retained and written cell contributions are shown separately;
  their sum is the updated cell. The complete arrays contain all coordinates.
  No coordinate is named as a semantic feature or claimed to store a color.
- **Reference prefixes versus generated prefixes:** the same six-color source
  is held-out test example 13 (zero-based). Full-vocabulary probability rows
  are shown under correct reference inputs and under the model's own selected
  earlier tokens. Argmax considers all nine classes, without filtering or
  temperature changes. Each row sums to one; visible probabilities are not
  renormalized. Orange boxes show the highest-probability candidate. Values
  at least 0.1 are annotated to two decimals; a displayed `1.00` is rounded.

The reference is `red red red black blue green EOS`. Both paths choose the
wrong fifth color, `green`. Their decoder inputs first differ at step 6:
the teacher supplies `blue`, whereas generation supplies its own `green`.
The freely generated result is `red red red black green blue EOS`. The
teacher-forced argmax row is a set of decisions using correct reference
prefixes, not a freely generated sequence. The reference is used to diagnose
the generation only after it has been produced.

`trace_lstm` exposes gates in PyTorch's input/forget/candidate/output block
order and verifies its replayed hidden and cell states against the native
`torch.nn.LSTM` kernel for every input prefix. The script also checks that
the two decoding paths have exactly equal logits before their input histories
diverge, that a one-step greedy loop agrees with scoring its selected prefix,
and that the reported probability rows normalize over all nine classes.
Forward passes are float32; softmax is evaluated in float64 from those logits.
Evaluation mode is used throughout observation; there is no optimizer update
or random sampling during inspection.

The output contains three PNGs, `ch02_trained_observations.npz` with all
observed arrays, and `TRAINED_FIGURES_RUN.json`. Training checkpoints and raw
arrays remain local. They can be reproduced using the commands above.
PyTorch's [versioned LSTM specification](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LSTM.html)
defines the parameter layout used to reconstruct these computations.
