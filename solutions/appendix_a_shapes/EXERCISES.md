# Appendix A | Exercises

Notation and Tensor Shapes

Exercises 1-4 establish the basics, 5-6 apply them, and 7-8 diagnose errors. Complete explanations and executable checks accompany every exercise in this directory.

## 1. Project each token

An input has four token rows and five features. Give the weight and output shapes for two output features per token. Why is the first output row independent of the other inputs?

## 2. Name the attention axes

$Q$, $K$, and $V$ have shapes $[3,2,4,5]$, $[3,2,7,5]$, and $[3,2,7,6]$. Give the transposed-key, score, and output shapes, and the normalized axis.

## 3. Separate two masks

Queries are at absolute positions $(3,4,5)$ and keys at $(2,3,4,5)$. Position 5 is padding. Give the combined causal/key mask and query validity; explain where each applies.

## 4. Count a cache

One layer, one example, and one KV head store two key and three value coordinates per token. Count scalars for a prefix of six and identify the growing axis.

## 5. Preserve token membership

Fill $[2,3,4]$ with integers 0 through 23. Split into two heads correctly and by direct reshape to $[2,2,3,2]$. Compare entry `[1,0,2,:]` and undo the correct split.

## 6. Catch silent broadcasting

Scores have shape $[2,1,2,2]$ and key-validity rows $(1,0)$ and $(1,1)$. Compare broadcasting the raw $[2,2]$ mask with inserting batch-preserving singleton axes.

## 7. Trace a weighted read

Use query $(1,0)$, keys $(1,0),(0,1),(1,1)$, and values $(10,0),(0,20),(30,30)$. Compute weights and output; mask the second key, then change its value.

## 8. Audit an axis port

A port uses `K.T` on $[3,2,7,5]$ and softmax over the query axis. Give the actual and intended transpose shapes and numerical tests even when lengths coincide.
