# Chapter 31 exercises

What Changed, and What Remains?

Exercises 1-4 work through counts and representations; 5-8 inspect Python stores;
9-12 design controlled evaluations. All have
[complete solutions](../../solutions/ch31_perspective/README.md).
Import helpers from `examples.ch31_perspective.model`.

For Exercise 2, selected examples from the history table are BERT (2018,
masked recovery), MQA (2019, shared KV heads), Longformer (2020, local/global
attention), and FlashAttention (2022, tiled attention execution).
Other correctly explained examples are valid.

Listing 31.3 runs the pair counts and delivery experiments:

```sh
python -m examples.ch31_perspective.run_examples --output output/ch31.json
```

1. **Count permitted pairs.** For a length-ten prefix, count all full causal query--key pairs. Then count pairs for a window of four, including the current position. Explain what changes for the earliest three queries.

2. **Locate an architectural change.** In the chapter's history table, find one change to the training task, one change to stored feature width, one change to the set of positions a query can read, and one change to attention execution. Explain the different purpose of each.

3. **Find two histories with the same summary.** Construct two streams with different first codes and different route updates that end at the same route. Describe the minimum information your summary needs to answer both of the chapter's questions. If any of 1000 codes can accompany any of four routes, find a lower bound on the number of bits in a fixed-length exact summary, ignoring metadata.

4. **Budget persistent history.** A hybrid has two full-attention layers with four stored scalars per position per layer, plus three recurrent layers with four state scalars each. Give the history count at prefix length $n$. Under a budget of 92 scalars, find the largest allowed $n$ for this accounting. Replace both full-attention layers by windows of four and derive the new bound. Convert both maximum history counts to bytes for two-byte scalars.

5. **Verify a window boundary.** Call `causal_pair_counts` at lengths 0, 3, and 5 with window three. List the ordered query--key pairs for length three, using positions numbered from one, and compare with both returned counts.

6. **Change the window size.** Use `make_history(517, ['X', 'Y', 'Z'])`. Call `compare_memory` with window sizes 2, 3, and 4. Inspect `first_code` and `current_route` under `recent_window` and explain the transition.

7. **Test a history with no route assignments.** Use `make_history(620, [])`. Compare all four rules at window size one. State what the output marks as unknown, and distinguish a missing event from an event that was discarded. Repeat with code 0 and explain why a valid zero must not be treated as `None`.

8. **Compare the storage records.** Run Listing 31.3. Select the one-update and eight-update records from `accuracy_sweep`. Compare `retained_units` and `unit` for each method. Explain why an event and a summary field should not be counted as identical bytes. State why unbounded Python integers and variable-length strings prevent a field count alone from proving bounded memory.

9. **Separate delay from interference.** On paper, design two groups of histories that test first-code recall. In one group, vary the number of unrelated messages. In the other, hold length fixed and vary the number of plausible competing codes. State which measurements would help explain a performance change.

10. **Compose an update and a retrieval.** On paper, design a delivery stream that stores several addresses and changes which delivery is active. Ask for the address of the final active delivery. Specify which history features a model needs and how to keep the answer from being predictable from the final event alone.

11. **Plan an ablation.** Compare a window-attention stack with a hybrid that replaces some layers with recurrent ones. Keep the window width fixed for the remaining attention layers. Define the task, the training and hardware controls, and the measurements for output quality, prefill, decoding, and history memory. State one limitation of the comparison.

12. **Transfer the evaluation idea.** Choose a forecasting, visual question, or robot policy example from the applications chapters. Identify one source of leakage or a misleading score. Design a paired test or controlled split that reveals the behavior you want to measure.
