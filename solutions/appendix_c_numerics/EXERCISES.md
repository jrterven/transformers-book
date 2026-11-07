# Appendix C | Exercises

Stable Numerical Computation

Exercises 1-4 establish the basics, 5-6 apply them, and 7-8 diagnose errors. Complete explanations and executable checks accompany every exercise in this directory.

## 1. Compare representable gaps

Under round-to-nearest, what happens to $1+2^{-8}$ in FP16 and in BF16 with ties to even?

## 2. Shift before exponentiating

For logits $(1000,1000+\log2,1000+\log5)$, give shifted exponentials, probabilities, and the middle-class loss.

## 3. Normalize only permitted keys

Normalize scores $(4,1,3)$ with only the last two keys allowed. Then remove every allowed key. What does the canonical attention routine specify?

## 4. Audit low-precision normalization

A proposed RMSNorm uses FP16 inputs, squares, and output. Choose a small input, a high-precision reference, and intermediates to inspect. What should change if squares become zero?

## 5. Keep the log loss finite

For logits $(0,-1000)$ and the second class as target, compare the rounded probability with the stable loss. Add 2000 to both logits and repeat.

## 6. Locate lost input information

Store logits $(1024,1024.25)$ in FP16. Compare softmax on the original float64 values, float64 on the stored values, and low-precision computation. Which discrepancy can a wider accumulator fix?

## 7. Place epsilon deliberately

Normalize $(10^{-4},-10^{-4})$ with epsilon $10^{-6}$ inside and outside the root. Check zero inputs. Why is the zero test insufficient?

## 8. Use absolute and relative tolerances

Reference values $(0,10^{-8},1000)$ are compared with $(10^{-9},1.1\times10^{-8},1000.01)$ using atol $2\times10^{-9}$ and rtol $2\times10^{-5}$. Evaluate combined, absolute-only, and relative-only checks; check NaNs separately.
