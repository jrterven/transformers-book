# Chapter 6 exercises

Training and Generating with a Small Transformer

Work through 1–7 as short traces and calculations, 8–11 as small programs,
and 12–15 as optional investigations. These statements match the manuscript.
Use evaluation mode for prediction comparisons and a numerical tolerance for
logits. [Complete solutions](../../solutions/ch06_training/README.md) cover
all fifteen exercises in the notebook and both PDF editions.

The vocabulary is PAD=0, BOS=1, EOS=2, then red=3, green=4, blue=5,
yellow=6, white=7, black=8. All commands and imports assume the repository root.

## 1. Follow what changes

For one training update, teacher-forced validation, and cached generation, state whether the weights change, where the target prefix comes from, and whether a loss is needed. Distinguish the roles of `zero_grad()`, `backward()`, `step()`, `eval()`, and `no_grad()`. Use a small model to check that computing gradients alone leaves its parameters unchanged, while an optimizer step can change them.

## 2. Align a padded batch

Write decoder inputs, labels, input-validity masks, and loss-selection masks for sources *red green blue* and *red blue*. Draw the causal availability grid. State how many target tokens contribute to the batch loss and explain why the diagonal is available. Use `collate` and `loss_sanity_example` to check the layout and count.

## 3. Calculate the signal

For probabilities $(0.8,0.15,0.05)$ with the first class correct, calculate the token loss and the derivative with respect to each logit. Then calculate the three-token mean and perplexity in the separate example whose correct-token probabilities at three positions are $(0.8,0.5,0.25)$. Distinguish the three-class example from the three-position example.

## 4. Aggregate before dividing

One batch contains three evaluated tokens with loss sum 1.2; another contains one evaluated token with loss sum 1.0. Calculate corpus NLL and perplexity. Compare with the unweighted mean of batch means. Use a fixed model in evaluation mode to evaluate the same dataset with two batch sizes. Keep one explicit generation cap in both calls and verify that corpus NLL agrees within numerical tolerance.

## 5. Limit a gradient

Apply a gradient-norm cap of 1 to the vector $(3,4)$. Calculate the original norm, scale factor, clipped vector, and new norm. Repeat with $(0.3,0.4)$. Explain why clipping preserves direction and why a cap on the gradient does not set Adam's learning rate or directly bound its parameter update.

## 6. Trace the input, the choice, and the cache

For a successful reversal of *red green blue*, list the consumed input, zero-based position, number of stored self-KV pairs per layer, and selected output at each call. Immediately after *blue* is selected, is its key already cached? Is the final selected `EOS` ever consumed in a single-example call that stops normally? Explain why each decoder layer stores its own keys and values but need not store old queries.

## 7. Account for memory

Calculate self-KV storage for $L=2$, $B=3$, $h=4$, $d_h=8$, $t=20$, and four bytes per element. Add static cross-KV storage for $S=6$. Report bytes and KiB, with $1 \mathrm{KiB}=1024$ bytes. Explain why $t$ is a count, and list at least three kinds of memory excluded from this calculation.

## 8. Test the stopping rule

Start from the stopping tests in `tests/test_training.py`, which temporarily replace `decoding.decode_prefix` with scripted logits. Use `greedy_decode(..., cached=False)` so that this replacement controls predictions. Make one stub emit `EOS` immediately and another emit *green* for a three-token cap without ending. Then make a two-example batch in which one row ends immediately and the other emits *green green EOS*. Write the expected returned arrays, including padding after a row finishes, before running the checks.

## 9. Overfit before scaling

Reuse `collate` and `train_step` to train repeatedly on four fixed source sequences. A small starting configuration is width 16, two heads, FFN width 32, one layer per stack, dropout zero, and Adam rate 0.01. Check both token loss and full generated answers as training proceeds. Introduce one deliberate error, such as an unshifted target input, by constructing a separate experimental `Batch`, and repeat from the same fresh initialization. Does a low teacher-forced loss still establish successful generation? Keep the canonical source unchanged.

## 10. Restore the same predictor

Save a small float64 model with `save_checkpoint`, including its configuration and vocabulary. Record logits on a fixed batch, change the live model's weights, then reload with `load_checkpoint`. Verify that the restored logits agree with the saved predictor and that the numerical precision and evaluation mode are restored. Explain why this inference checkpoint is insufficient to resume the same Adam training trajectory exactly.

## 11. Probe the limits of the task

Load the selected checkpoint and evaluate by length within the trained range. Create a separate length-extrapolation set with `make_splits(min_length=7, max_length=8,` `n_train=8, n_validation=8, n_test=64)` and use its test split. Change no weights or selection rules. Pass `generation_max_new_tokens=10` explicitly in both the in-range and extrapolation evaluations; eight colors need nine output tokens including `EOS`. Report the two settings separately and inspect substitutions, repetitions, and stopping errors.

## 12. Separate validation from testing (optional experiment)

Write a comparison loop using `train_step` and `evaluate` on the training and validation splits. Try two learning rates, select a checkpoint using validation NLL, then evaluate the selected model on the test split once. The complete `run_experiment` script automatically calls the test set at the end, so use the component functions while making this comparison. Report the selection rule, selected update, token loss, exact match, and termination rate. Keep the validation split and generation cap the same for both learning rates.

## 13. Detect information leakage (optional experiment)

Compare fresh causal and noncausal models with the same initialization and sampled training batches. Temporarily replace `build_masks` in both the Chapter 5 model module and Chapter 6's `decoding` module with a wrapper that keeps source and target key validity but allows future target keys. Use `unittest.mock.patch` context managers so both bindings are restored automatically; the solution provides the wrapper.

For the noncausal model, measure loss using `token_loss`. Generate with `cached=False`. Do not use the ordinary `evaluate` helper, which requests cached generation. Also score the same reference labels one prefix at a time, cutting away future inputs before each prediction. Compare whole-reference NLL, prefix-only NLL, and generated exact match. Explain why the noncausal cache assumption fails and why a later color input reveals an earlier label, while the final `EOS` label has no revealing input.

## 14. Break and repair a cache (optional experiment)

Start with a small float64 model in evaluation mode, two decoder layers, and a padded source. At every prefix length, compare `CachedDecoder.step` with the last logits row from `decode_prefix`. If the reference prefix has padded rows, replace padding inputs by repeated `EOS` in both paths; the incremental step rejects `PAD`. In a temporary subclass, reset the positional offset to zero before each step. Then repeat with a fresh instance of the original class. Identify the first failing prefix and explain why testing only the first token misses the error.

## 15. Time equal work (optional experiment)

Use `benchmark(model, source, prefix, repeats=25)` on a CPU model in evaluation mode. Supply a fixed, nonpadding prefix with at least two tokens; the helper times the same final input token in both paths. Report the thread count and arithmetic precision. Separate source encoding and cached target prefill from the later token calculation, and repeat the measurement. Explain how early `EOS`, different continuations, or including source encoding in only one path would distort the comparison.
