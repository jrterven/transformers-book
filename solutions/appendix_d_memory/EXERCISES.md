# Appendix D | Exercises

Complexity and Memory Accounting

Exercises 1-4 establish the basics, 5-6 apply them, and 7-8 diagnose errors. Complete explanations and executable checks accompany every exercise in this directory.

## 1. Count permitted work

For six causal tokens, window width three including the current token, two query heads, and key/value widths four each, count permitted pairs and the MACs for the two attention products when excluded pairs are skipped.

## 2. Track trainable arrays

A model stores two million FP32 weights, with only 50,000 trainable. Count all weights plus FP32 gradients and two Adam moments for the trainable subset. Name another allocation to measure.

## 3. Convert hybrid history to bytes

Two attention layers each have one KV head with key/value widths eight. One recurrent layer stores 32 scalars per sequence. Batch size is two and every scalar uses two bytes. Count history for lengths 16 and 32.

## 4. Define a fair timing run

Two models generate equally long responses to the same prompt. Define separate prompt and decode timers, controlled settings, and throughput numerators.

## 5. Count occupied and reserved slots

Lengths are 3 and 7. One layer and one KV head use key/value widths four and two-byte scalars. Count packed payload, rectangular allocation, and capacity eight per sequence. Then add one token to each sequence.

## 6. Solve a history budget

One sequence has two attention layers with one KV head, key/value widths eight and two-byte scalars, plus 32 FP32 recurrent scalars. Find the largest full-cache prefix under 1,152 bytes and the plateau with a window of eight.

## 7. Count live overlap

Persistent arrays use 8 MiB, saved activations 3, a forward workspace 5, and a backward workspace 4. The four phases are forward workspace live, forward workspace freed, backward workspace live, and saved tensors freed. Reproduce totals and add a 2 MiB copy through phases 1-3. What if it is a view?

## 8. Audit a throughput claim

Four concurrent requests start at 0 and each emit tokens at 90, 115, 140, and 165 ms. Find TTFT, inter-token latency, aggregate decode throughput, and aggregate complete-request output throughput. Audit $16/0.075$ and a claim that halving KV bytes halves time.
