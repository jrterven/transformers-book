# Appendix B | Exercises

Matrix Calculus and Optimization

Exercises 1-4 establish the basics, 5-6 apply them, and 7-8 diagnose errors. Complete explanations and executable checks accompany every exercise in this directory.

## 1. Follow a scalar derivative

For $z=\theta x$, $L=(z-y)^2/2$, take $\theta=1$, $x=4$, $y=6$. Find loss and derivative, and the direction of a small descent step.

## 2. Differentiate cross-entropy

For logits $(0,\log3)$ and the first class as target, compute probabilities, loss, gradient, and the sum of gradient entries.

## 3. Weight by valid tokens

Two pieces contain two and five valid labels, with mean gradients 4 and $-3$. Find the combined mean. Does splitting the second piece change it when sums are used?

## 4. Clip once and separate decay

Two tensors have gradients $(6,8)$ and $(0,0)$. Clip their global norm to 5. Separately, evaluate AdamW for old parameter 2, corrected moments 1 and 4, learning rate 0.1, decay 0.2, negligible epsilon.

## 5. Check every matrix derivative

Use $X=((2,1),(1,3))$, $W=((1,0,-1),(2,1,0))$, and $G=((1,0,-1),(0,2,1))$. For $L=\sum G_{ij}(XW)_{ij}$, check all weight and input derivatives by float64 central differences at $10^{-6}$. Why hold $G$ fixed?

## 6. Add shared contributions

For $Y=XW+b$ with the same $G$, compute the shared-bias gradient. A second branch contributes ones to the input gradient. Find the total and explain why it is a sum.

## 7. Order accumulation and clipping

Equal-size pieces have scalar mean gradients 10 and $-6$. Clip at 5. Compare clipping before and after accumulation; repeat with scale 128, and test clipping before unscaling.

## 8. Trace the first AdamW step

Start at parameter 2, zero moments, beta values 0.9 and 0.99, learning rate 0.1 and decay 0.2. Compute the first step for gradients 3 and $-3$, neglecting epsilon.
