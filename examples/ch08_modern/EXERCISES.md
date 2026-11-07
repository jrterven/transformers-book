# Chapter 8: exercises

From the Original Block to Modern Decoders

Exercises 1-6 build understanding by hand; 7-11 use the Python model; 12-14 are optional investigations. Use CPU float64, evaluation mode, and fixed weights for logit comparisons. All 14 worked solutions are in [the solution guide](../../solutions/ch08_modern/README.md).

Import model components from `examples.ch08_modern.model` and `build_masks` from `examples.ch05_transformer.model`.

## 1. Move the normalizer

Take an input $(3,4)$ and a sublayer that always returns a zero update. Use LayerNorm with unit gains, zero offsets, and $\epsilon=0$ in both arrangements. Find the post-norm and pre-norm outputs. Trace the path followed by the input in each case.

## 2. Rescale the same token

Compute RMSNorm on $(3,4)$ and $(6,8)$ with $\epsilon=0$ and gains $(2,1/2)$. Show where the gain enters the calculation. Explain why the two scaled inputs give the same result, and why the mean square after the gains can differ from one. With unit gains, find the output's Euclidean length. Finally, compare LayerNorm and RMSNorm on $(3,4)$ and $(13,14)$ using unit gains, zero offsets, and $\epsilon=0$. Which operation removes a shared additive shift?

## 3. Change the sign of a feature

Use $\mathbf{u}=(3,-1)$ and the down projection from the worked example. Replace $\mathbf{g}=(0,2)$ with $(-1,2)$. Calculate both SiLU values, their products with $\mathbf{u}$, and the four coordinates after the down projection.

## 4. Match a parameter budget

A bias-free two-projection feed-forward layer has model width 12 and intermediate width 48. Count its weights. Find the SwiGLU width that gives the same count. Repeat for model width 10 and conventional intermediate width 40. Find the ideal SwiGLU width, then round it up to a multiple of eight.

## 5. Check the sign in RoPE

Use $\mathbf{q}=(1,0)^\top$, $\mathbf{k}=(0,1)^\top$, $m=3$, $n=1$, and $\omega=\pi/6$. Compute the dot product by rotating both vectors, then compute it with $R((n-m)\omega)$. Repeat after a shared shift of two positions. Explain how this test exposes a sign error that the equal-vector example can hide.

## 6. Build a head from pairs

Take $d_h=4$, base 10,000, and unrotated query and key $(1,0,1,0)^\top$. At $m=3$ and $n=1$, find both frequencies, both relative angles, the two dot-product contributions, their sum, and the scaled attention logit. Repeat for $n=3$. Explain why these logits are not yet attention probabilities and why RoPE alone would still allow a score for $n>m$.

## 7. Inspect a small denominator

Use `rms_norm` with unit gains, `eps=1e-6`, and float64 rows $(3\cdot10^{-4},4\cdot10^{-4})$ and $(0,0)$. Compute the output mean square of the first row by hand and check it in Python. Multiply just that row by ten and repeat the measurement. Confirm that the zero row remains finite.

## 8. Rebuild the worked gate

Build `SwiGLU(4, 2)` and call `.double()`. Zero the weights in `gate_proj` and `up_proj`. Set the first column of each weight matrix to $(0,2)^\top$ and $(3,-1)^\top$, respectively. Copy $W_{\mathrm{down}}^\top$ into `down_proj.weight`. Use `torch.no_grad()` for the assignments. Check the output on the float64 input $(1,0,0,0)$.

## 9. Probe the attention mask

Set seed 81 and create one `ModernDecoder().double().eval()`. Use `torch.long` tensors with a batch axis for `[7, 8, 9, 10]` and `[7, 8, 9, 16]`. Compare the first three logit rows, then inspect the change at the final row. Append `PAD=0` to the first sequence and compare its four original rows with the padded run. Use `torch.testing.assert_close` with `rtol=0` and `atol=1e-10`. Obtain the allowed-key grid from Chapter 5's `build_masks(valid, valid)`, where `valid = ids != PAD`. Compare its last diagonal entry with a causal-only lower triangle. Why would the appended-padding comparison alone miss a missing padding-key mask?

## 10. Check a full run

Run the Chapter 8 module and record its output shape, parameter count, and gradient check. Derive the 5,744 parameters from the dimensions. Predict how many parameters a third block adds, and confirm that count by creating a model with `n_layers=3`. Identify the operation that mixes positions and the one that expands the features within each position.

## 11. Move the origin, then change the context

Reuse the fixed model from Exercise 9. Compare logits for `[7, 8, 9, 10]` at `start_pos=0` and `start_pos=7`, using `atol=1e-10`. Explain why equality follows through both blocks. Next prepend token 16 and compare the last four rows with the original four. Measure the difference and explain why shared-shift invariance no longer requires equality. State what this experiment does and does not establish about long contexts.

## 12. Catch a reset position (optional)

Use `rotate_pairs` on four copies of $(1,0)$ with shape $[1,1,4,2]$ at positions 0 through 3. The default frequency for this one pair is one radian per position. Rotate the keys again in two slices at indices $[0,1]$ and $[2,3]$, then concatenate with `dim=-2`. Compare both versions. Next reset the second slice to indices $[0,1]$. Compare the unscaled scores against a query $(1,0)$ rotated at position 3. This models the position calculation for a key that will enter a cache.

## 13. Isolate a residual gradient (optional)

Use a differentiable input $(1,2,4)$, a zero update, and LayerNorm with unit gains, zero offsets, and $\epsilon=10^{-6}$ in both placements. Use a fresh float64 tensor for each calculation. First sum the three output coordinates and compare input gradients. Then use the loss $y_1-y_2$, where $y_1,y_2$ are the first two output coordinates, and repeat. Explain the direct residual contribution and why the first post-norm result is not evidence that post-norm always blocks gradients. Check the second post-norm gradient with central finite differences using step $10^{-5}$ and absolute tolerance $10^{-8}$.

## 14. Design a matched comparison (optional)

Plan a next-token experiment comparing a ReLU feed-forward layer with SwiGLU. Choose the widths to give similar numbers of feed-forward weights. Keep the rest of the block fixed. Specify the data split, update budget, optimizer, and seed plan. Compare loss per valid token on held-out data. Describe what you would measure separately to compare execution time.
