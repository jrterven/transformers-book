# Appendix E | Exercises

Convolution and Parallel Scans

Exercises 1-4 establish the basics, 5-6 apply them, and 7-8 diagnose errors. Complete explanations and executable checks accompany every exercise in this directory.

## 1. Keep the kernel order

For inputs $(3,1,-2)$ and lag coefficients $(2,-1)$, compute causal outputs. Give weights for an oldest-to-newest cross-correlation window with left padding.

## 2. Retain the initial state

For $h_t=h_{t-1}/2+x_t$, initial state 6, and inputs $(0,2,1)$, find the final state by stepping and unrolling. Separate initial-state and input contributions.

## 3. Understand a reset map

Compose $(1/2,2)$ followed by $(0,5)$ and then reverse their order. Explain what happens to the incoming state and first write.

## 4. Prevent circular wraparound

Run the direct and padded-FFT helpers on input $(0,0,1)$ and kernel $(1,2,3)$. Compare their three outputs, and explain what wraps if the FFT length is three.

## 5. Build an independent causal reference

Use the lower-triangular Toeplitz matrix with diagonal 2 and first subdiagonal $-1$ on $(3,1,-2)$. Replace only the final input by 100. Which outputs may change? Check the padded FFT too.

## 6. Verify every prefix

Use maps $(1/2,2)$, $(1/4,-1)$, $(3/4,1)$ and initial state 4. Check all inclusive prefixes with `compose_diagonal`, both final parenthesizations, and reversed time order.

## 7. Find the limit of a fixed kernel

Let $h_t=a_t h_{t-1}+x_t$, zero initial state, and retentions $(1/2,1/4,3/4)$. Compare impulses $(1,0,0)$ and $(0,1,0)$. Can one lag-one kernel coefficient represent both?

## 8. Separate work, span, and rounding

For eight maps compare final-only tree reduction, standard upsweep/downsweep exclusive scan, and independently computed sequential prefixes. Reproduce the binary64 cancellation example and explain the numerical limit of associativity.
