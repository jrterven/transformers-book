# Chapter 26. Models for Code

Run the examples from the repository root with Python.

```sh
python -m examples.ch26_code_models.run_examples --output output/ch26.json
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

See the [11 exercises](EXERCISES.md) and [complete solutions](../../solutions/ch26_code_models/README.md).
