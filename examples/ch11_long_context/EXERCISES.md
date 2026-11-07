# Chapter 11 exercises

Long Context, Sparse Access, and Sequence Compression

Exercises 1–4 develop intuition by hand; 5–8 use the NumPy examples; 9 accounts for costs; 10 combines causal behavior and normalization; 11–12 are open-ended investigations. Import NumPy as `np` and use functions from `examples.ch11_long_context.model`. Positions in 1–4 start at one; indices in 5–8 and 10 start at zero.

For cost accounting, a stored main-attention KV entry is one key and its associated value. Count indexer scores separately from entries read by main attention. A summary and a raw entry count separately even when they cover the same source token. Index features, partial-summary state, and scratch buffers are additional storage.

1. **Count the grid.** A document has six tokens. Draw its full bidirectional grid, its full causal grid, and its bidirectional local grid at radius one. Label the query and key axes and count the allowed pairs. In the decoder's last row, which positions are available under full causal attention?

2. **Trace a relay.** Use ten positions numbered 1 through 10 and radius one. Position 5 is global. Trace how information from position 2 can reach position 9 in two layers. How many layers would the shortest local-only path need? Explain why the relay uses outputs from consecutive layers. Does this route remain valid under a causal mask? If position 1 were the only global position instead, could it relay information from position 2 in the same way?

3. **Choose, then weight.** The index scores for four positions are $(1.4,0.6,1.0,1.9)$. Select the top two. In descending order of index score, their main scores are $(0,\log 2)$ and their values are $(3,0)$ and $(0,3)$. Compute the attention weights and output.

4. **Pool one coordinate.** A summary receives features $(4,0,2)$ and weights $(0.25,0.5,0.25)$. Compute the result. Find another triple with the same weighted sum. Explain what a query can infer from that coordinate alone.

5. **Inspect a global mask.** Call `local_mask` with length 7, radius 1, and `global_positions=(3,)`. List the readable keys for zero-based queries 0, 3, and 6. Repeat with `causal=True`. Identify the sets that change and explain why.

6. **Filter before gathering.** Use query `np.array([1.0])`. Take key rows $0,1,2,3$, value rows $10,20,30,40$, and index scores $(0.8,1.1,0.6,9.0)$. Give the keys and values a shape of $[4,1]$. Call `selected_attention` with $k=2$ and pass `[True, True, True, False]` as `allowed`. Report the selected indices, weights, and output. Repeat with $k=5$ and explain which rows were added.

7. **Compare summary windows.** Put values $1$ through $6$ in a feature array of shape $[6,1]$. Use a matching array of zeros for the gate logits. Call `block_summaries` with stride two, first without overlap and then with overlap. Compare the end positions and summary values. Explain how the start of the sequence affects the first window.

8. **Give coordinates different gates.** Use feature rows $(2,0)$, $(0,4)$, and $(6,2)$. Set the first gate column to $\log(1,2,1)$ and the second to $\log(3,1,2)$, taking the logarithm of each entry. Call `block_summaries` with stride three. Compute each summary coordinate by hand and compare the result with Python.

9. **Account for three designs.** At a decoding step with 128 visible positions, compare dense attention, selection of eight raw entries after a full index scan, and a read of all stride-eight summaries plus the four most recent raw entries. Using the accounting convention above, count stored main-attention KV entries, index scores, and main reads. Count each raw entry and each summary separately, even if they cover the same position.

10. **Check visibility and joint normalization.** Put values 1 through 9 in a feature array of shape $[9,1]$ and create zero gate logits with `np.zeros_like`. Use stride three without overlap. Place the query at zero-based position 7. Use the end positions to identify which summaries it can read. Change the last feature from 9 to 90 and build the summaries again. Check which summaries change and which remain visible. Explain why checking the start position could let a query read a future feature. Finally, select only the first visible summary and combine it with raw positions 6 and 7. Give all three entries zero main logits. Compare a single softmax over their values with averaging the separately normalized summary and recent-window outputs.

11. **Measure missed evidence (optional).** Design a synthetic test with one target row and several distractors. Hold the main query, keys, and values fixed. Vary the target's index score so it moves above and below competing rows, and choose several values of $k$. For each case, say how you would record whether the target is selected and how far the output is from full attention.

12. **Plan a long-document comparison (optional).** Compare dense attention, local windows, learned selection, and sequence summaries on two tasks. One task asks for a precise fact from far back in a document. The other asks for facts from different passages to be combined. Choose context lengths and evidence positions and give each system the same inputs. Specify how you would measure answer quality, cache memory, prefill time, and decoding time.

[Full worked solutions](../../solutions/ch11_long_context/README.md) cover every exercise.
