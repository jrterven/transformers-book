# Chapter 10 exercises

Incremental Decoding and KV Representations

Exercises 1-4 are hand calculations, 5-8 are Python investigations, and 9-12 are optional extensions. Run Python from the repository root and import NumPy as `np`. Here `model`, `run_examples`, and `latent` refer to modules inside `examples.ch10_kv`. All 12 [worked solutions](../../solutions/ch10_kv/README.md) are available in the executed notebook and both PDF editions.

1. **Follow the first two predictions.** The prompt is *the red* and the next two sampled tokens are *car* and *stopped*. After prefill, which row's logits are used to sample *car*? In the next step, identify the input token, the new query's zero-based position, and all the positions it may read. Name the vectors that the layer keeps for the step after that. If generation stops after sampling *stopped*, is that token already represented in the cache?

2. **Count rows and comparisons.** A prompt has three tokens. After prefill the decoder processes two more one-token inputs. For those two steps, count the query rows per head and layer under full-prefix recomputation and cached decoding. Then count the allowed causal query–key pairs for each method. Explain why the cached method still needs more than one comparison per step. If prefill supplies the first sampled output, how many outputs have been sampled after these two subsequent calls?

3. **Ask two questions of one history.** Two query heads share keys $(1,0)$ and $(0,1)$ and values $(2,0)$ and $(0,1)$. Use queries $(1,0)$ and $(0,1)$ at width two. Calculate the scaled scores, softmax weights, and output for each head. Explain why the outputs can differ.

4. **Count a decoder's cache.** Take two sequences, each with 1,024 stored positions, and a 12-layer decoder with 12 query heads. Key and value widths are 64 and the cache uses two bytes per scalar. Find the total cache in MiB with 12, 3, and 1 KV heads. How many query heads share a KV head in each case?

5. **Build a mask for a chunk.** Use `causal_mask` in `model` to get the mask for two queries starting at absolute position 2, with four key positions starting at zero. Repeat with a query start of zero and identify the reads lost by resetting the position. Now suppose there are two sequences and four query heads. What mask shape lets each sequence have its own read pattern and share it across its heads?

6. **Compare full and cached attention.** Call `projected_example()` in `run_examples` for the six-position $q$, $k$, and $v$ arrays. Keep all four query heads and slice $k$ and $v$ to their first two heads with `[:, :2]`. Call `grouped_attention` with a full causal mask to get the output and weights. Then call `run_chunks` on the same arrays with sizes $(3,2,1)$ to get the output and final cache. Compare the outputs with `np.testing.assert_allclose` using `rtol=0` and `atol=1e-12`. Check that the cached keys and values match the supplied $k$ and $v$. Increase the value at position 4 by 100, rerun the same chunks, and verify that outputs at positions 0–3 are unchanged. Explain why this checks causality more directly than merely matching two implementations.

7. **Check what a byte count means.** In Exercise 6, sum the cache arrays' `nbytes`. Predict the same number with `cache_bytes`, using one layer and the cache dtype's `itemsize`. Cast $k$ and $v$ to float32 and run `run_chunks` again. Confirm the new cache count. Report the dtype of each cache array and the dtype of the attention output.

8. **Compare two orders of latent attention.** In `latent`, call `latent_inputs()` to get the input tuple. Pass it with `*inputs` to both `expanded_attention` and `latent_attention`. Compare the output and weight arrays with `np.testing.assert_allclose`. For the first head, reconstruct the two scaled scores using the content and rotary contributions. Count the scalars in the latent and rotary key arrays, the third and fourth entries of the input tuple.

9. **Spend a cache budget (optional).** With 24 layers, 16 query heads, key and value widths of 64, and two bytes per scalar, the MHA example uses 192 MiB for one sequence with 2,048 positions. Suppose the budget for the KV arrays is 192 MiB. How many equally long sequences could fit with four KV heads? With one KV head? If there is one sequence and four KV heads, how many positions fit? Treat the model's supported context length and other memory costs as separate limits.

10. **Catch a position reset (optional).** Use one pair of coordinates and a RoPE angle of $\pi/2$ per position. An earlier key and a new query both start as $(1,0)$. The key is at position 0 and the query at position 1. Compute their unscaled dot product with correct rotations. Repeat with the query accidentally rotated as if its position were 0. Explain why preserving the cache rows alone is insufficient.

11. **Choose the scale after absorption (optional).** An MLA layer has content query and key width 4, rotary width 2, and latent width 9. Find the denominator of the expanded score. Find the length of the transformed content query and state which denominator to use after absorption. How would using $\sqrt{d_c+d_R}$ affect the softmax?

12. **Design an inference comparison (optional).** Plan a comparison between full-prefix recomputation and a cache on the same causal decoder. Give an input and chunk schedule for checking numerical equality. Choose a prompt length, batch size, continuation length, data type, and device for timing. Replay the same continuation tokens in both runs. State which work you would include in the prompt and decoding measurements, and how you would report cache bytes and temporary memory separately.
