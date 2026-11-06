# Chapter 4 exercises: Self-Attention from Vectors to Matrices

Exercises 1–8 develop interpretation and hand calculations; 9–13 check dimensions
and information flow with Python; 14–15 are optional calculus challenges.
Keep full precision until the final display. All 15 worked solutions are in the
[solution notebook and PDFs](../../solutions/ch04_attention/README.md).

Unless stated otherwise, use unmasked attention and these teaching values:

```python
H = [[1, 0, 1, 0], [0, 1, 0, 2], [1, 1, 1, 1]]
W_Q = W_K = [[1, 0], [0, 1], [0, 0], [0, 0]]  # separate parameters
W_V = [[0, 0], [0, 0], [1, 0], [0, 1]]
Q = K = [[1, 0], [0, 1], [1, 1]]
V = [[1, 0], [0, 2], [1, 1]]
```

Rows correspond to *the*, *red*, and *car*. The code uses NumPy arrays and zero-based
indices, so mathematical position 3 is index 2. Compute `S = Q @ K.T / sqrt(2)`,
normalize each row to get `A`, and return `O = A @ V`. A Boolean mask uses `True`
for allowed keys. Run functions from `examples.ch04_attention.run_examples` and
`examples.ch04_attention.attention` inside the repository.

1. **Identify the jobs.** For the three-token example, state the shapes and roles of $H$, $W^Q$, $W^K$, $W^V$, $Q$, $K$, $V$, $A$, and $O$. Which objects are learned parameters shared across positions, and which are computed for the current input? Does $HW^V$ combine token positions? Does a row of $A$ give probabilities over the vocabulary?

2. **A query with equal scores.** Replace the first query by $(0,0)$ and leave the keys and values unchanged. Calculate its unmasked weights and output. Explain why a zero score still gives a nonzero weight.

3. **Follow another query.** Use $\mathbf q_2=(0,1)$ to calculate three dot products, three scaled scores, the weights, and the separate weighted value vectors. Add those vectors and compare with the second row of $O$. Run `one_query_example(1)` to check each intermediate result. Which sum runs over features, and which sum runs over positions?

4. **Change what is read.** Keep $Q$ and $K$ unchanged, but replace the third value by $(3,-1)$. Recalculate the first output. Which intermediate matrices change, and which remain the same? Derive the change $\Delta\mathbf o_i$ for any query in terms of $A_{i3}$. Explain why the attention map alone does not determine the output.

5. **Use two sequence lengths.** Let $Q$ have shape $4\times3$, $K$ shape $6\times3$, and $V$ shape $6\times5$. Write the shapes of the scores, attention weights, and output. Explain why each output still corresponds to a query position. Which dimension determines the scaling divisor?

6. **Inspect directionality.** Use the unrounded example to compare $A_{13}$ with $A_{31}$. Explain how symmetric raw scores can produce asymmetric normalized weights.

7. **Separate scaling from shifting.** Start from the raw score row $(1,0,1)$. Calculate softmax before scaling, after division by $\sqrt{2}$, and after adding $1000$ to every scaled score. Use stable softmax for the last case. Which operations change the weights, their ordering, and the output obtained with the original $V$? Explain your results using score differences.

8. **Mask at the wrong stage.** Apply ordinary softmax to the first *scaled* score row, $(1/\sqrt{2},0,1/\sqrt{2})$, then set its future weights to zero without renormalizing. Compare the resulting output with the correct causal output $(1,0)$. Would setting the future scores to zero before softmax exclude those keys? Explain.

9. **Follow the axes.** For $B=2$, $h=4$, $n_q=3$, $n_k=5$, $d_k=8$, and $d_v=6$, write every shape in the attention calculation. Then deliberately use `k.T` and explain why the result is wrong.

10. **Combine padding and causality.** Repeat the toy $Q$, $K$, and $V$ across two examples and one head, giving each tensor shape $[2,1,3,2]$. Use valid lengths three and two with *right padding*. Create a Boolean NumPy array `key_valid` with rows `[True, True, True]` and `[True, True, False]`. Keep the toy vectors unchanged, including the vector at the padded position, so the mask's effect is easy to isolate. Combine the shared $3\times3$ causal mask with `key_valid[:, None, None, :]`, of shape $[2,1,1,3]$. Verify that no query reads the padded key. Inspect the padded query's output and explain why key masking alone need not zero it. Why would also masking out that entire query row trigger an error in our implementation?

11. **Test the future-information rule.** Copy $H$ and add $(2,-1,3,-2)$ to its third row only. Recompute $Q$, $K$, and $V$ with the original projections. Check that the first two causal output rows are unchanged and equal a run on the original two-position prefix. Compare with unmasked attention. Explain why a causal mask cannot guarantee this property if earlier input rows already contain information from future positions.

12. **Keep keys and values paired.** Keep $Q$ fixed and reorder the rows of both $K$ and $V$ using the index array `[2, 0, 1]`. Compare the unmasked outputs and explain the result by following the attention columns. Repeat with a causal mask, reordering the original mask's columns with the keys. This represents the original permissions in a new storage order; constructing a new lower triangle would impose different permissions. Finally, reorder only $K$ and explain why the original read is generally lost.

13. **Separate order from parallel computation.** Without a mask or position information, permute all input rows using the same index array and check that the output rows move with them. Explain why the operation does not assign an absolute position to a token. Then explain why all queries within one layer can be evaluated together, even though the next layer and the next generation step still depend on earlier computation.

14. **Choose a finite-difference step (advanced).** Use the demonstration's `gradient_max_error` helper with its optional `step` argument. Use the toy $Q$, $K$, and $V$, the causal mask, and incoming gradient with rows $(0.2,-0.4)$, $(0.5,0.3)$, and $(-0.1,0.6)$. Compare maximum absolute errors for $\varepsilon\in\{10^{-2},10^{-4},10^{-6},10^{-8}\}$, holding the inputs, mask, and incoming gradient fixed. Why can a step that is too large or too small make the estimate less accurate?

15. **Learn through the projections (advanced).** Extend the derivatives through $Q=HW^Q$, $K=HW^K$, and $V=HW^V$. Reuse the toy $H$ and projections, but set `W_K[0, 0] = 0.5`, so the query and key weights have different numerical entries. Use the causal mask and an incoming gradient $G$ with rows $(0.2,-0.4)$, $(0.5,0.3)$, and $(-0.1,0.6)$, as in the demonstration. Derive gradients for $H$, $W^Q$, $W^K$, and $W^V$, adding all three paths into $H$, and check each with central differences. As a starting point, differentiating $Q=HW^Q$ gives $\partial L/\partial W^Q=H^\top(\partial L/\partial Q)$. Treat $W^Q$ and $W^K$ as separate variables throughout.
