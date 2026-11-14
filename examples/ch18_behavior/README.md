# Interpreting and Aligning Model Behavior

The address desk proposes one of three actions for fictional requests. A reads
the saved address, B starts an update, and C asks for clarification. The
program reads language-model scores and never connects to an account.

[Exercises](EXERCISES.md) · [Worked solutions](../../solutions/ch18_behavior/)

## Redraw and verify

Run these commands from the repository root in an isolated Python environment.
The checked execution used Python 3.12, the pinned versions below, and a CPU.
The figure script reads the [recorded run](PRETRAINED_RUN.json). It needs
Matplotlib and NumPy but does not download weights.

```sh
python -m pip install -r requirements-ch18-models.txt
python -m examples.ch18_behavior.make_figures \
  --record examples/ch18_behavior/PRETRAINED_RUN.json --output output/ch18-figures
python -m unittest discover -s tests -p 'test_model_behavior.py' -v
```

## Repeat model inference

```sh
python -m examples.ch18_behavior.run_pretrained --output output/ch18/PRETRAINED_RUN.json
```

The first run downloads the Apache-2.0
[Qwen3.5-0.8B-Base checkpoint](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68)
at revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. Pass
`--checkpoint /path/to/snapshot` to reuse the pinned files offline. The
downloaded weights occupy about 1.65 GB. Float32 CPU execution needs additional
memory for expanded parameters and temporary arrays. The script verifies the
weights, configuration, and tokenizer file hashes and checks exact checkpoint
loading. It uses four CPU threads, text inputs, deterministic PyTorch
operations, evaluation mode, and native reference kernels. Numerical last
digits can vary with the runtime.

The three patching pairs differ at exactly one token each. The first covers
24 blocks, five sites, and two swap directions (240 interventions). The other
two use five blocks at the same five relative sites (50 interventions each).
Each measured block has two self-patches, and the pre-edit site and final
readout provide further controls. The program saves logits for A, B, and C
and reports the changed B-minus-A gap in natural-log units. We use layer
numbers and token positions starting from one in the record.

Each of the three routing groups contains six authored cases, two for each
action. All policies see the same unmodified logits. We compare the free
first token, the best first token among the three labels, and the same
three-label rule with deferral for selected reads or updates below a
restricted probability of 0.80. The first two agree on these cases and
get 9/18 correct. The last gets 11/18, with all six appropriate clarifications
and seven unnecessary ones. It correctly routes zero of six updates,
compared with three of six without the threshold. The threshold is an
illustration on a fixed teaching set. The record includes both the restricted
scores and the total mass of the three candidates in the full distribution.
