# Chapter 31. What Changed, and What Remains?

Run the fixed experiments from the repository root with Python.

```sh
python -m examples.ch31_perspective.run_examples --output output/ch31.json
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
[complete solutions](../../solutions/ch31_perspective/README.md).
