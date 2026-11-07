# Chapter 15 exercises

Delta Rules and Hybrid Stacks

Exercises 1-4 use hand calculations, 5-8 use NumPy and `examples.ch15_hybrids.model`, and 9-10 investigate interference and evaluation. Exercises 8-10 are optional. All ten [worked solutions](../../solutions/ch15_hybrids/README.md) appear in each edition.

The update first forms $\widetilde S=\alpha S$, prediction $\widehat v=\widetilde S k$, and error $e=v-\widehat v$. It then writes $S_{\rm new}=\widetilde S+\beta e k^\top$ and reads $y=S_{\rm new}q$. Vectors are columns and the state has shape `[value width, key width]`.

For a unit key, Equation 15.3 gives $S_{\rm new}k=(1-\beta)\widehat v+\beta v$. Equation 15.5 gives the read change from the correction: $S_{\rm new}q-\widetilde S q=\beta e(k^\top q)$. Equation 15.6 writes the same update as $S_{\rm new}=\alpha S(I-\beta kk^\top)+\beta vk^\top$.

The Jamba-v0.1 schedule in Figure 15.1 repeats an eight-layer block four times. Attention is at position 5 of each block; the other mixers are Mamba. The feed-forward sublayers at positions 2, 4, 6, and 8 use MoE (2 of 16 experts); the others are dense. Layer numbers start at one.

1. **Read the Jamba schedule.** Number the 32 layers starting at one. Use Figure 15.1 to list the layers with a growing KV cache. Count how many layers have a recurrent mixer followed by an MoE feed-forward sublayer. For layer 14, identify the mixer, the feed-forward type, and whether the token is routed to experts. Explain why selecting two of 16 experts does not mean using exactly $2/16$ of all model parameters.

2. **Count bytes across different stores.** A three-layer stack has one full-attention cache with key width three and value width two. Its recurrent layers store 12 and 8 scalars per sequence. At a prefix of seven tokens, calculate the persistent history in bytes when cached keys and values use two bytes per scalar and the recurrent states use four. Repeat at a prefix of 14 tokens.

3. **Correct one column.** Start from $S=\left(\begin{smallmatrix}1&2\\0&3\end{smallmatrix}\right)$, choose key and query $(0,1)^\top$, value $(4,-1)^\top$, decay $\alpha=0.5$, and correction $\beta=0.25$. Compute the decayed prediction, the error, the new state, and the output. Explain what happens to the first column.

4. **Trace a sign change.** Use key $(1,0)^\top$, a zero write value, and $S=I$ in Equation 15.6. Set $\alpha=0.6$ and compare $\beta=0.5$, $\beta=1$, and $\beta=1.5$. Start from $S=I$ each time. Read the resulting state with both coordinate unit vectors and describe what changes along the key direction. Separately, use Equation 15.3 with scalar decayed prediction 2, target 5, and $\beta=1.5$. Explain how the result relates to the negative carry coefficient.

5. **Check a read with Python.** Use `delta_step` with the arrays and gates from Exercise 3. The function returns the output and the updated state. Repeat from the original state with query $(1,0)^\top$ and all other inputs unchanged. Use the state columns to explain the change in the output. Then set $\alpha=\beta=1$ and apply the same unit-key write twice. Verify that the second write leaves the state unchanged. Repeat with key $(0,2)^\top$ and explain why the unit-key overwrite result no longer applies.

6. **Watch three stores grow.** Use schedule `("delta", "attention", "delta")` and call `initial_stores`. Feed tokens $(1,2)$, $(0,1)$, and $(-1,2)$ through `stack_step`, unpacking each result as `trace, stores`. Pass those stores into the next step. Use `history_counts` after each step. Inspect the shapes of the two recurrent state arrays and of the key and value arrays in the attention store.

7. **Change the middle mixer.** Compare the run in Exercise 6 with the schedule that uses delta updates in all three layers. Start each schedule with new empty stores. The trace starts with the input at index zero. Compare the final token's trace at indices one and three across the schedules. Explain why later representations can change even if the first and last mixer types are the same.

8. **Check the cache against a full pass (optional).** Use schedule `("attention",)` and three two-feature tokens of your choice. Build the queries, keys, and values with this chapter's projection rule. Use `scaled_dot_product_attention` from Chapter 4's `attention.py`. Give it a Boolean lower-triangular `allowed_mask` with the diagonal included. Compare its output rows with the last trace entry from each streaming `stack_step`. Change only the third input and verify that the first two outputs remain unchanged in both calculations.

9. **Write to overlapping directions (optional).** Start from a zero state with one value coordinate and two key coordinates. Take $\alpha=\beta=1$. Write value 2 at key $(1,0)^\top$. Then write value 5 at key $(1/\sqrt{2},1/\sqrt{2})^\top$. Read the final state at both keys. Restart from zero and repeat with second key $(0,1)^\top$. In each run, compare the read at the first key before and after the second write. Explain any change using Equation 15.5. In the overlapping run, write value 2 at the first key once more and compute what the second key now reads.

10. **Plan a controlled comparison (optional).** Design sequences with an early table of delivery addresses and later messages that move the active index, such as advancing it by one in a circular list. Ask for the address at a specified index, the current index, and the address at the current index. Explain how to compare an attention-only, recurrent-only, and hybrid model at several test lengths. Identify which training and evaluation settings to match and which per-sequence memory quantity to report. Vary the table size separately from the number of updates. Explain why a small fixed table and circular moves alone do not establish a general separation between architectures. As a harder extension, replace the moves with swaps of two index labels and give an example where reversing two swaps changes the final index.
