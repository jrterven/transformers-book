# Chapter 5 exercises

Multiple Heads, Position, and the Complete Architecture

Exercises 1–7 develop intuition and short calculations; 8–13 apply the
components; 14–15 are optional programming challenges. Predict the result
before running each experiment. Use float64 tensors, `.double()` modules,
and evaluation mode for exact comparisons. Evaluation mode still permits
gradients. Import `torch.nn.functional as F` where needed.

Use the complete [canonical modules](model.py), the
[worked examples](run_examples.py), and the [chapter guide](README.md).
All fifteen exercises have matching [written, notebook, and PDF solutions](../../solutions/ch05_transformer/README.md).

1. **Follow one prediction.** We translate *the red car* into *el coche rojo*. At the target position predicting *coche*, name the target inputs that may be read and the source words that may be read. Explain the jobs of target self-attention, cross-attention, the FFN, and the vocabulary projection. Which operations exchange information between positions? Which softmax distributes weights over source positions, and which distributes probabilities over vocabulary entries?

2. **Follow the head axes.** Create a tensor `x` of shape $[1,2,4]$ whose two token rows are $(1,2,3,4)$ and $(5,6,7,8)$. Construct `attention = MultiHeadAttention(4, 2)` and apply `attention.split_heads(x)`. Write the two token rows assigned to each head, then use `attention.merge_heads` to recover the original tensor. Check the entries as well as the shape: both the token and head axes have length two here, so the shape alone cannot identify an incorrect interchange. Explain why reshaping alone, without moving the head axis, gives the axes a different meaning.

3. **Project before splitting.** Take a row $x=(1,2,3,4)$ and a row-convention query matrix $W^Q$ equal to $I_4$ except for an additional 1 in row 4, column 1 (counting from one). Compute $xW^Q$, then split its four coordinates between two heads. Change only $x_4$ from 4 to 9 and repeat. Does head 1 change? Explain why a head's projected width does not limit it to the corresponding segment of the original input. Check with `q_proj` and `split_heads`, remembering the stored-weight transpose.

4. **Compare three attention shapes.** For batch size two, four heads, source length five, and target length three, give the shapes of encoder, decoder, and cross-attention maps. Name what each axis counts. Explain why a triangular source mask would impose an unnecessary restriction on cross-attention in this translation setting.

5. **Move a token.** Compute the width-four positional vector at position 2. Combine it with the embedding $(0,1,0,2)$ using the original embedding scale. Compare with the same embedding at position 1. Identify which terms remain fixed when the token moves.

6. **Inspect a post-norm wrapper.** Let the incoming vector be $(1,2,3,4)$ and the sublayer update be zero. Write the residual sum, then apply LayerNorm with unit scale, zero offset, and $\varepsilon=10^{-5}$. Explain why the wrapper does not return the original vector. Repeat with input $(3,3,3,3)$ and the same zero update. You can check both results with `nn.LayerNorm(4, eps=1e-5).double()`.

7. **Track dropout values.** Use a floating-point tensor for $(1,2,3,4)$ and apply dropout with probability $1/2$ in training mode. List the two possible outputs for each coordinate and calculate its expected value. What is the output in evaluation mode? Explain why entering a `torch.no_grad()` block does not by itself give evaluation behavior.

8. **Follow both heads.** Use the projections in the two-head setup below to calculate the second query's raw scores, attention weights, and output in head 2. Start by writing its projected query, keys, and values. Concatenate its result with head 1 and apply $W^O$. Check the results against the corresponding row returned by `toy_multihead`. Explain why equal first rows in the two attention maps do not imply equal maps everywhere.

9. **Separate concatenation from mixing.** Replace $W^O$ by the identity: call `toy_multihead` with its optional `output_projection` argument set to `torch.eye(4)`. The optional argument uses the book's row-vector matrix convention. Which returned quantities change? Then count the bias-free attention parameters for $d_{\mathrm{model}}=8$ with two heads and with four heads. Keep total model width fixed and include all three input projections and the output projection.

10. **Test the positionwise FFN.** Reproduce the hand-chosen FFN output for $(1,-2,0,3)$. Reuse the parameter assignments in `toy_feed_forward` to construct your own `PositionwiseFeedForward(4, 8).double()` module; the helper itself returns numerical results. Place several vectors in different rows and verify that permuting those rows permutes the outputs in the same way. Change one input row and check that other FFN outputs remain unchanged.

11. **Read a vocabulary distribution.** Recalculate the three logits and probabilities in the three-entry readout setup below. Use *coche* as the reference next token and compute its negative log-probability at full precision. Verify it with `F.cross_entropy(logits, torch.tensor([1]))`, where `logits` has shape $[1,3]$. Explain why index 1 selects the reference vocabulary entry, rather than a source position or an attention head.

12. **Count the predictions that matter.** Use the labels from `example_batch()` with a tensor of zero logits of shape $[2,4,12]$. How many labels contribute to the loss, and what probability does each receive? Calculate the mean loss with `ignore_index=0`. Compare it with averaging all eight entries returned by `reduction="none"`. Explain the difference. Set `requires_grad=True` on the logits and check that the ignored row has zero loss gradient. Does ignoring a PAD label remove the PAD vocabulary class from the denominator at the other rows?

13. **Keep query validity separate from key validity.** Recreate the one-layer model and `example_batch()` used in `full_model_example`, and call `model.eval()`. Save the source-validity mask before changing any IDs. Replace IDs at padded source positions by other in-range IDs, pass the saved mask explicitly, and compare valid target logits before and after the change. Inspect padded target logits and explain why their values are excluded from the loss rather than forced to zero by a key mask.

14. **Break causality deliberately (optional challenge).** Temporarily replace the module's `build_masks` function with a wrapper that returns the original source mask and `target_valid[:, None, None, :]` for the target mask. Use `unittest.mock.patch` as a context manager so the original function is restored automatically; the solution demonstrates the setup. Do not change any model weights. This removes the triangle while retaining target key validity. Recreate the seed-7 one-layer example in evaluation mode, change `target_ids[0, 2]` to 6, and compare the first two target positions' logits before and after that change. Leave the temporary context to restore the original causal mask and repeat. Explain why the comparison changes, and confirm that `check_prefix_and_padding` rejects the version with the broken mask.

15. **Build a deeper model (optional challenge).** Construct a fresh model with `n_encoder_layers=2` and `n_decoder_layers=2`; the existing `full_model_example()` still constructs one layer per stack. Keep the remaining small-model settings and reuse `example_batch()`. Check that corresponding parameters in adjacent layers are different Python objects. In this model, their first-element addresses, obtained with `data_ptr()`, should also differ. Equal initial values, such as LayerNorm's scales and offsets, do not imply shared parameters. Run `check_prefix_and_padding` on this model, then reuse the masked loss and backward calculation from the one-layer example. Check finite gradients and explain why every decoder layer reads the final encoder memory.

## Numerical setups

For Exercises 8–9 use row vectors with

$$
H=\begin{bmatrix}1&0&1&0\\0&1&0&2\\1&1&1&1\end{bmatrix},
\quad W^Q=W^K=I_4,
$$

$$
W^V=\begin{bmatrix}0&0&1&0\\0&0&0&1\\1&0&0&0\\0&1&0&0\end{bmatrix},
\quad W^O=\begin{bmatrix}1&0&1/2&0\\0&1&0&1/2\\1/2&0&1&0\\0&1/2&0&1\end{bmatrix}.
$$

Project first, then take the first two projected coordinates for head 1 and
the last two for head 2. Divide each head's dot products by $\sqrt{2}$.
The unmasked toy helper returns `(output, weights, concatenation)`.

For Exercise 10, the FFN uses $W_1=[I_4\;-I_4]$ and
$W_2=[I_4;I_4]$ (vertical stacking), zero biases, and ReLU between the maps.
For Exercise 11 use $y=(1,-1,0,2)$, zero output bias, vocabulary order
`el, coche, EOS`, and

$$
W_{\mathrm{vocab}}=\begin{bmatrix}1&0&-1\\0&-1&0\\0&0&1\\1/2&0&1/2\end{bmatrix}.
$$

The complete seed-7 example uses `Transformer(12, 12, d_model=4, n_heads=2,
d_ff=8, n_encoder_layers=1, n_decoder_layers=1).double().eval()` and
`example_batch()`. These randomly initialized parameters are separate from
the hand-chosen numerical setups above.
