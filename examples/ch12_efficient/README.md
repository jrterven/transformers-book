# Chapter 12. Efficient Execution

Run from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch12_efficient.run_examples --output output/ch12.json
python -m unittest discover -s tests -p test_efficient.py -v
```

## Carry softmax state across tiles

`online_update` keeps a maximum, denominator, and weighted numerator for
each query row. For `Q` query rows and value width `Dv`, initialize the
state with a length-`Q` array of negative infinities, a length-`Q` array
of zeros, and a `[Q, Dv]` array of zeros. Pass that state and one tile of
scores and values to each update. When a tile introduces a larger maximum,
the update rescales both accumulated sums before adding the new terms.
Rows with no allowed keys in a tile keep their previous state.

The hand example uses scores `(0, log(2), log(3))` and values
`(1, 2, 4)`. An extra masked score between the two tiles leaves the
state unchanged. The final numerator divided by the denominator is
`17 / 6`.

## Check a complete tiled read

`tiled_attention` takes one head's projected arrays `q[Q, Dk]`,
`k[K, Dk]`, and `v[K, Dv]`. It returns `[Q, Dv]` and uses float64
arithmetic. Tile sizes can differ and can exceed the sequence lengths.
With `causal=True`, each query uses its absolute position beginning at
`query_start`. Keys begin at zero. The causal query positions must all
fall within the retained key prefix. The function skips wholly masked
tiles and masks individual entries in the remaining tiles.

The command compares tiled output with Chapter 4's materialized
attention on the same arrays, both unmasked and causal. Tile edges may
change the order of floating-point operations, so the tests compare
outputs within a numerical tolerance.

## Count a score array and a score tile

`score_workspace` returns element counts for one full score array and
the largest score tile allowed by the dimensions. Multiply either count
by the element size to get bytes for that array alone. Peak memory also
depends on values, masks, accumulators, other temporary arrays, and the
output. The NumPy code illustrates the calculation order of tiled
attention. GPU kernels also account for data movement and device-specific
execution.
