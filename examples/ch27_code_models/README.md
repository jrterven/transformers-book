# Chapter 27. Models for Code

Run the examples from the repository root with Python.

```sh
python -m examples.ch27_code_models.run_examples --output output/ch27.json
python -m unittest discover -s tests -p test_code_models.py -v
```

## Test the behavior of fixed candidates

The task is to decide whether two half-open integer intervals share
a position. Equal endpoints describe an empty interval. The
two smoke cases cover an interior overlap and separated intervals.
The four edge cases cover touching endpoints, an empty interval,
an overlap of length one, and negative endpoints.

The command calls four functions defined in the chapter package.
`score_candidate` compares each result with the expected Python
Boolean. It reports individual outcomes, including wrong output
types and exceptions, and counts a candidate as correct when it
passes every case. All four candidates pass the smoke cases, and
two pass the complete suite. These results describe the supplied
functions and tests.

## Compare samples and inspect infilling format

`pass_at_k` reports the probability that a uniformly chosen subset
of `k` among `n` candidates includes a passing candidate. The function
subtracts the number of all-failing subsets from the total before
dividing. With four candidates and two passing the complete suite,
`pass_at_k(4, 2, 2)` gives `5/6`.

`serialize_fim` places a code prefix and suffix between generic
markers, followed by a middle marker. The string exposes the
input layout of a fill-in-the-middle example. For checkpoint use,
consult its tokenizer and format documentation for the actual
special tokens and their order.

## Keep the denominator visible

The JSON `smoke` group contains two original checks and `edges` contains
four added checks. Together they form the six-case complete suite.
Across four candidates, 21/24 individual checks pass, but only 2/4 candidates
pass every check. These are different questions.

The pass@k helper counts uniform subsets of one fixed candidate pool.
Its interpretation as an unbiased estimator for fresh attempts assumes
independent draws from the same procedure. Compute it per task before averaging;
pooling candidates from unrelated tasks changes the experiment.
Coverage alone does not describe the answer chosen by a selector.

Fill-in-the-middle uses a causal decoder on a reordered prompt: the supplied
suffix appears before generation of the missing body. The generic markers
show the layout; actual checkpoint formats must match their tokenizer.

See the [11 exercises](EXERCISES.md) and [complete solutions](../../solutions/ch27_code_models/README.md).

## Read the measured demonstrations

The [records](records/) contain the actual prompts, model output, decoding
settings, checkpoint revisions, and per-case results from local inference.
The fixtures and expected results are in [demo_tasks.py](demo_tasks.py).

| Task | Checkpoint | Result on the chapter fixture |
| --- | --- | --- |
| Fill the missing `overlaps` body | Qwen2.5-Coder-1.5B | The same completion passes 3/6 half-open and 6/6 closed-interval checks. |
| Generate `peak_bookings` | Qwen3.8-27B | 6/8 complete passes, with 54 checks per tested function. |
| Generate `peak_bookings` | Gemma 4 26B-A4B IT | 5/8 complete passes on the same 54 checks. |
| Generate `peak_bookings` | LFM2.5-24B-A2B | All eight requests hit the short 1,024-token limit inside the reasoning block. |
| Repair `conflicts` | Qwen3.8-27B and Gemma 4 26B-A4B IT | The original passes 4/8. The issue-only replacements pass 6/8 and 7/8. Adding the helper definition gives 8/8 for each. |

The FIM record includes two complete greedy infills and a separate probability
probe with the same partial return expression and three suffixes. Its
probabilities are over the whole vocabulary. For generation, each seed starts
a fresh request. An output that reaches the token limit is recorded as an
incomplete attempt. Repair uses greedy decoding for Qwen and Gemma. The two
repair prompts add the helper's source only in the second condition. The test
process has the same helper in both conditions. These are behavioral checks,
so an implementation can pass individual checks while ignoring a requested API.

## Rebuild the figures from the recorded runs

From the repository root, install the plotting dependencies and run the figure
script. The three listings are extracted from the same JSON files as the plots.

```sh
python -m pip install matplotlib==3.11.2
python -m examples.ch27_code_models.make_figures \
  --records examples/ch27_code_models/records \
  --output output/ch27/figures --examples-output output/ch27/listings
```

## Repeat the local model runs

The measured runs used Python 3.12 and Apple silicon. The evaluator uses the
macOS subprocess profile in `demo_tasks.py` to run one submitted function per
attempt. The runner downloads the pinned checkpoints on the first run and
verifies the weight shards. Use `--offline` after caching the snapshots, or
`--model-root` with directories named for the model keys.

```sh
python -m pip install -r requirements-ch27-models.txt
for model in fim15 qwen38 gemma26 liquid24; do
  python -m examples.ch27_code_models.run_pretrained \
    --model "$model" --output "output/ch27/runs/$model.json"
done
```

The LFM row uses the same output-token cap and generation sampler as the
Qwen and Gemma rows. Qwen and Gemma run with their optional thinking modes
disabled. The recorded results apply to these prompts, quantized checkpoints,
and limits.
