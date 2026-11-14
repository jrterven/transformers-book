# Chapter 18 exercises

Interpreting and Aligning Model Behavior

Exercises 1–4 work through internal measurements and their controls. Exercises
5–8 move from scores to decisions and the evidence needed to test them.
[Worked solutions and a notebook](../../solutions/ch18_behavior/) are available.

1. **Recover the odds.** A read prompt gives logits $z_A=4$ and $z_B=2$. A patch changes them to $z_A=3$ and $z_B=4$. Compute both gaps and odds of $B$ relative to $A$. If $z_C=6$ after the patch, which of $A,B,C$ would be chosen?

2. **Measure recovery.** Use recipient, donor and patched gaps $(-2,2,3)$. Compute $\Delta$ and $R$. Repeat for $(-2,-2,3)$. Explain how you would report the second result.

3. **Read a feature and change it.** Let a probe use the first coordinate of $\mathbf h=(2,1)$ and the model use the second to compute $g=-2h_2$. Change the first coordinate to $-1$. What changes? Propose a second intervention to test whether the model uses $h_2$.

4. **Check a patch.** Explain why swapping a vector before the only differing token should have near-zero effect in a causal model. What result do you expect from patching an earlier position after the final block? Check both predictions on the first pair using the saved run.

5. **Compare two rewards.** On intended actions $A,B,C,A$, compare proposals $A,A,A,A$ with $A,B,C,\varnothing$. Compute the fraction that have valid labels and the fraction that choose the right action.

6. **Apply a threshold.** Give the three actions logits $(2,1,0)$ in order $A,B,C$. Normalize over the three candidates. Use `route_policy` to compare the unthresholded decision with the $0.80$ rule. If the reference action is $A$, what failure did the second rule introduce?

7. **Test a shortcut.** Design two new pairs of requests that separate the word *change* from the desired update action. Explain how you would split wording families across training and testing and how you would report errors on ambiguous requests.

8. **Change the decision cost.** In the expected-cost example, double the cost of asking on a clear request. Recompute all three expected costs. Which action now minimizes cost? State the additional evidence you would need to use these costs and probabilities in a real desk.
