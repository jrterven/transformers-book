# Chapter 9 exercises

Scaling Data, Compute, and Expert Capacity

Exercises 1-6 are hand calculations, 7-11 are Python investigations, and 12-14 are optional extensions. Run code from the repository root. Import the helpers from `examples.ch09_scaling.model`; `teaching_weights` is in `examples.ch09_scaling.run_examples`.

All 14 [worked solutions](../../solutions/ch09_scaling/README.md) are available in the executed notebook and both PDF editions.

1. **Spend the same budget.** Estimate the compute for a 12-billion-parameter dense model trained on 300 billion tokens. At the same budget, how many training tokens can a 24-billion-parameter model process? How does the budget change if we instead double both the original model size and the original token count?

2. **Count what the model sees.** Corpus A has 120 million token positions and is used for three complete passes. Corpus B has 360 million positions and is used for one pass. Compute $D$ for each. With the same dense model, compare their estimates from $C\approx6ND$. Explain why their held-out losses may differ.

3. **Trace a token.** Use $\mathbf{x}=(1,0,0,0)$ and routing probabilities $(0.1,0.6,0.2,0.1)$. Expert 2 returns $(2,-1,0,3)$ and expert 3 returns $(0,1,4,-1)$. Find the expert update and residual output for Switch. Repeat with the two-expert gate rule and explain why the denominator is $0.6+0.2$.

4. **Count a wider bank.** A routed branch has model width 6, expert hidden width 12, and four experts. Use two bias-free matrices per expert and one bias-free router. Count the stored weights and the weights used by one accepted token under top-one routing. What changes under top-two routing when both experts accept the token?

5. **Separate demand from room.** Eight rows request experts $1,2,2,2,2,2,3,4$. Use $T=8$, $E=4$, and capacity factor $\phi=1.5$ in $K_{\rm cap}=\lceil\phi T/E\rceil$. Admit requests in row order. Find the capacity, accepted counts, overflow rows, overflow fraction, and fraction of reserved slots used. Explain how overflow can occur while slots are empty. Repeat the two fractions at capacity five.

6. **Read a router diagnostic carefully.** Four rows each have logits $(0,0,0,0)$. Break ties in favor of the lowest expert ID and give each expert capacity one. Find each row's probabilities, the request fractions $f$, the mean probabilities $P$, and the raw balance term $b$. Which rows are accepted? Compare this with four rows that request different experts with probability $0.7$ for their winner and $0.1$ for each other expert. Does $b=1$ prove that requests are balanced?

7. **Refine the sweep.** Use `allocation_sweep` at $2.4\times10^{22}$ FLOPs and test model sizes of 10, 12, 13, 14, and 20 billion. Find the best row and explain what changed from the chapter's coarse grid of 5, 10, 20, 40, and 80 billion parameters. Use `training_flops` to verify the budget for that row after converting both counts back from billions.

8. **Check the score shift.** Create a NumPy array of shape $[1,4]$ containing the logs of $(0.1,0.6,0.2,0.1)$. Run `route_top1` at capacity one. Add 1,000 to all four logits and run it again. Check that the selected ID is the same and that the probabilities and gates agree within rounding error. Use `numpy.testing.assert_allclose` for the floating-point comparisons. Explain both results.

9. **Move the capacity limit.** Make a score array of shape $[8,4]$ filled with zeros. Put a 2 in the winning column of each row, using indices `[0, 1, 1, 1, 1, 1, 2, 3]`. Run `route_top1` at capacities 2, 3, and 5. Count accepted rows per expert and report which one-based rows overflow. Explain why empty slots in other experts do not change those results.

10. **Check the residual.** Call `teaching_weights()` from this chapter's `run_examples` module. It returns the router weights, the expert up projections, and the expert down projections, in that order. Put $(1,0,0,0)$ in a NumPy array of shape $[1,4]$ and multiply it by the router weights. At capacities one and zero, pass the scores to `route_top1` and use its IDs, gates, and mask in `expert_branch`. Compare the branch updates and the outputs after adding the input.

11. **Compare requests with probabilities.** Build the eight-token input by taking rows `[1, 0, 0, 0, 0, 0, 2, 3]` from `np.eye(4)`, where `np` is NumPy. Use the router weights from Exercise 10 and call `route_top1` with capacity three. Count the requests per expert and divide by eight to find $f$. Average the probability matrix across rows to get $P$. Use $b=E\sum_i f_iP_i$ to compute the raw balance term. Repeat at capacity two and report which statistics change. Use the mask to count accepted rows at each capacity.

12. **Track two devices (optional).** All eight input rows start on device A. Device A holds experts 1 and 3, and device B holds experts 2 and 4. Use the requests and capacity-three rule from Exercise 5. Count how many accepted rows each device processes. With width 1,024 and two bytes per coordinate, how many bytes cross between devices for the forward dispatch and return, counting vector payload only?

13. **Find the best continuous allocation (optional).** Set $Q=N_B D_B=4000$, so $D_B=Q/N_B$. In $\ell_{\rm toy}=1+0.4/\sqrt{N_B}+2/\sqrt{D_B}$, the two penalties have the form $a/z+bz$ for $z=\sqrt{N_B}$. Use differentiation, or the arithmetic--geometric mean inequality, to show that the penalties are equal at the minimum. Find $N_B$ and $D_B$ there and predict the effect of quadrupling $C$. Check both predictions with `allocation_sweep` using sizes from 5 to 40 billion in increments of 0.1 billion.

14. **Plan a fair comparison (optional).** Write a plan to compare a dense feed-forward decoder with a top-one routed decoder. Keep the attention architecture, tokenizer, optimizer, and data split fixed. Decide whether to match stored weights or approximate training compute, and say what you would count. Explain how you will measure loss per held-out token and the fraction of requests that overflow. For timing, use the same hardware, batch size, prompt length, and number of generated tokens. State whether you include the time to process the prompt.
