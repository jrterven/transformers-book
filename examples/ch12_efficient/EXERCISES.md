# Chapter 12 exercises

Efficient Execution

Exercises 1-4 use hand calculations; 5-8 investigate the NumPy code. Exercises 9-11 are optional extensions on stability, bottlenecks, and measurement. Use natural logarithms and one MiB = $2^{20}$ bytes. For Python work, import NumPy as `np` and use `examples.ch12_efficient.model`. All eleven [worked solutions](../../solutions/ch12_efficient/README.md) appear in each edition.

1. **Draw a response timeline.** A cached decoder receives a four-token prompt and emits a five-token response. Its prefill takes 80 milliseconds and each subsequent decoding pass takes 12 milliseconds. Mark the time when each output token is selected. Give the time to first token and the total model execution time. Ignore sampling and serving overhead.

2. **Compare two batches.** Use the times from Exercise 1 for the single request. Then consider three requests batched together, each emitting five tokens, with a batched prefill of 100 milliseconds and subsequent passes of 24 milliseconds each. Compute aggregate output throughput and per-request completion time for each setup. Say which measurement answers a user waiting for one response.

3. **Count one intermediate.** Take 2048 queries and keys per head, 16 heads, and two bytes per score. Calculate the size of the full score arrays for one layer and one sequence. Repeat for four bytes per score. List two memory costs that this calculation leaves out.

4. **Follow a changing maximum.** The first tile has scores $(0,\ln 3)$ and scalar values $(2,4)$. The second tile has score $\ln 6$ and value $1$. Record $m$, $\ell$, and $u$ after each tile. Compare the final result with direct softmax over all three scores. Then put the second tile first and trace where the rescaling factor changes. Finally, average the two independently normalized tile outputs. Explain why that average disagrees with the complete result.

5. **Check an uneven grid.** Call `score_workspace` for 5 queries, 11 keys, 2 query rows per tile, and 4 key columns per tile, in that argument order. Draw the grid and mark the partial tiles. Explain why adding the sizes of all tiles and finding the size of the largest tile describe different quantities.

6. **Change the tile shape.** Use one-feature query rows $(1,2,3)$, key rows $(0,1,-1,2)$, and value rows $(4,1,3,2)$. Store them as arrays of shapes $[3,1]$, $[4,1]$, and $[4,1]$. Run `tiled_attention` without a mask for tile shapes $(1,1)$, $(2,3)$, and $(3,4)$. Compare the three outputs using `np.testing.assert_allclose`. Use `score_workspace` to compare the full-grid and largest-tile element counts.

7. **Pass through an empty tile.** Initialize a one-row state for `online_update` with `np.array([-np.inf])`, `np.zeros(1)`, and `np.zeros((1, 1))`. Supply scores $[[4,7]]$ and values $[[8],[5]]$. Set both entries of a Boolean mask of shape $[1,2]$ to `False`. Record the state. Reuse that state with scores $[[2]]$, values $[[9]]$, and no mask. Inspect the denominator before dividing and explain when the output becomes defined.

8. **Trace a rectangular causal mask.** Make zero query and key arrays of shapes $[2,1]$ and $[5,1]$. Make a value array of shape $[5,1]$ with rows $(1,3,7,2,9)$. Call `tiled_attention` with query tiles of size one, key tiles of size two, and `causal=True`. Compare `query_start=0` with `query_start=3`. For each call, list the keys each row can read and explain the difference in the outputs.

9. **Stress the exponentials (optional).** Give one query the score row $(800,799,801)$ with scalar values $(1,3,5)$. Use the empty state from Exercise 7. Feed `online_update` score arrays of shapes $[1,2]$ and $[1,1]$ and value arrays of shapes $[2,1]$ and $[1,1]$. Verify that the state after the final tile is finite. Restart with every score increased by 50. Explain what changes in the state and what should agree within tolerance.

10. **Explain a changing bottleneck (optional).** A fused attention kernel reduces prefill time on a long prompt. The same change has little effect on the time for one cached decoding step at batch size one. For a prompt of length $n$, draw the score grid at prefill and for the first cached pass after it. Include the current position's key in that pass. Use the shapes to give a possible explanation. Name two measurements that would help you test your explanation. For an illustrative 100 ms decoding step with 20 ms spent in attention, calculate the new step time if only attention becomes twice as fast. What is the overall speedup?

11. **Plan a fair kernel comparison (optional).** Design a comparison of materializing and fused causal attention in prefill. Choose three prompt lengths, one fixed batch size, and one precision. Say what you would keep fixed, how you would check output agreement, when you would start and stop timing, and which memory statistic you would collect. Add one measurement that would help you tell whether a speed difference came from moving fewer bytes.
