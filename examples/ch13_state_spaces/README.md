# Chapter 13. Structured State Spaces and Long Convolutions

Run from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch13_state_spaces.run_examples --output output/ch13.json
python -m unittest discover -s tests -p test_state_spaces.py -v
```

## Read one system as a scan or a filter

The examples use one input and output channel and a state of width `N`.
Pass the input sequence as a vector `u`, the transition as an `[N, N]`
matrix, and `b_bar` and `c` as vectors of length `N`. These vectors
represent the column write map and row read map from the book.

`ssm_scan` reads the inputs in order and returns the outputs and a
trace of the states. Row zero of the trace is the state after reading
the first input. The initial state defaults to zero. An ordinary
streaming step only needs the current state, while the full trace helps
check these small examples.

`ssm_kernel` starts at `b_bar`, reads it with `c`, and applies the
transition once between consecutive filter taps. With zero initial
state and fixed coefficients, the resulting causal convolution agrees
with the scan. A nonzero initial state adds its own response to the
input-driven convolution. The scalar fixture uses `x0=[2]` to show
this contribution.

## Follow the causal convolution

`causal_convolution` multiplies the first available filter taps by
the input positions in reverse order, starting at the current input.
`fft_causal_convolution` uses a transform length of
`len(u) + len(kernel) - 1` for linear convolution and returns the
first `len(u)` output positions. The example compares this result with
a short circular convolution, where tail terms wrap to earlier
positions.

## Discretize a system and gate a convolution

`bilinear_discretize` takes the continuous transition `a`, the input
vector `b`, and a positive scalar step `delta`. It solves the two
linear systems with the same left-hand matrix and returns the
discrete transition and input map.

`hyena_stage` multiplies by an input gate, applies the causal filter,
and multiplies by an output gate. Both gate vectors have one entry
per input position. The function returns all three intermediate
sequences so the effect of each gate can be inspected.

## Exercises and complete solutions

The [eleven exercises](EXERCISES.md) progress from hand calculations to
NumPy investigations and optional extensions. Every solution appears in
the [executed notebook and both PDFs](../../solutions/ch13_state_spaces/README.md).
The code's saved state trace uses `n * N` values; the streaming recurrence
carries only `N` state values between steps. The supplied gate arrays are
controlled teaching inputs. Gate-producing operations in an autoregressive
model must also be causal.
