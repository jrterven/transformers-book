# Chapter 14. Selective State Spaces

Run from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch14_selective_ssm.run_examples --output output/ch14.json
python -m unittest discover -s tests -p test_selective_ssm.py -v
```

## Precompute the update coefficients

`zoh_coefficients` takes positive scalar or array step sizes for the
scalar system `dx/dt = -x + u`. It returns the retain factor and the
complementary write factor. The use of `expm1` keeps the write factor
visible when the step is so small that the retain factor rounds to one.

`selective_scan` takes a scalar input sequence `u[T]`, effective write
and read arrays `b[T, N]` and `c[T, N]`, and the diagonal retain factors.
Use `a[T, N]` for coordinate-specific factors or `a[T, 1]` for one factor
shared across coordinates at each step. The initial state `x0[N]`
defaults to zeros. The trace records the state after each input. For
the restricted scalar system, `b` is the write factor returned by
`zoh_coefficients` and `c` is one.

`compose_diagonal` applies one affine update followed by another. Its
arguments are pairs `(a, r)`, where `r` already includes the input for
that step. The returned pair represents both steps as a single map.

## Compare three forms of one SSD calculation

The SSD examples use one retain factor per time step. `ssd_recurrent`
uses the diagonal scan with the same retain factor on every coordinate.
`ssd_dense` builds the causal coefficient matrix, multiplies by the
input sequence, and adds the response to the initial state. The matrix
returned by `ssd_dense` contains the input-driven coefficients.

`ssd_chunked` builds a matrix only for the current chunk. It adds the
readout from the incoming state, scaled by each position's prefix
product of retain factors. The last row of the local decay matrix
accumulates that chunk's writes into the outgoing state. The function
returns the output sequence and the final state. The last chunk can be
shorter than the chosen chunk size.

The fixed four-position fixture follows the book. A second fixture
uses unequal retain factors, a nonzero initial state, and a shorter
final chunk. A retain factor of zero removes the previous state's
contribution at that step while the new write remains. The examples
use precomputed arrays and float64 arithmetic to expose the update
rules of simplified SSM cores.

## Exercises and complete solutions

The [ten exercises](EXERCISES.md) progress from hand calculations to NumPy
investigations and a controlled memory-task design. Every solution appears
in the [executed notebook and both PDFs](../../solutions/ch14_selective_ssm/README.md).

The exact zero retention used in the reset fixture is a discrete test input.
A finite positive step has strictly positive exponential retention in exact
arithmetic. The scan's saved trajectory stores `n * N` state values; streaming
the simplified core carries only the last `N` values. A complete block also
retains its short convolution buffer. Fixed-size state can store both first
and latest facts when its update is designed to preserve them.
