# Chapter 18 exercises

Interpreting and Aligning Model Behavior

The 16 exercises build from hand calculations and tests of saved Qwen outputs
to feedback objectives, policy decisions, and an audit design.
[Worked solutions and a notebook](../../solutions/ch18_behavior/) are available.

1. **Recover the odds.** A read prompt gives logits $z_A=4$ and $z_B=2$. A patch changes them to $z_A=3$ and $z_B=4$. Compute both gaps and odds of $B$ relative to $A$. If $z_C=6$ after the patch, which of $A,B,C$ would be chosen?

2. **Measure recovery.** Use recipient, donor and patched gaps $(-2,2,3)$. Compute $\Delta$ and $R$. Repeat for $(-2,-2,3)$. Explain how you would report the second result.

3. **Read a feature and change it.** Let a probe use the first coordinate of $\mathbf h=(2,1)$ and the model use the second to compute $g=-2h_2$. Change the first coordinate to $-1$. What changes? Propose a second intervention to test whether the model uses $h_2$.

4. **Check a patch.** Explain why swapping a vector before the only differing token should have near-zero effect in a causal model. What result do you expect from patching an earlier position after the final block? Check both predictions on the first pair using the saved run.

5. **Compare two rewards.** On intended actions $A,B,C,A$, compare proposals $A,A,A,A$ with $A,B,C,\varnothing$. Compute the fraction that have valid labels and the fraction that choose the right action.

6. **Apply a threshold.** Give the three actions logits $(2,1,0)$ in order $A,B,C$. Normalize over the three candidates. Use `route_policy` to compare the unthresholded decision with the $0.80$ rule. If the reference action is $A$, what failure did the second rule introduce?

7. **Test a shortcut.** Design two new pairs of requests that separate the word *change* from the desired update action. Explain how you would split wording families across training and testing and how you would report errors on ambiguous requests.

8. **Change the decision cost.** In the expected-cost example, double the cost of asking on a clear request. Recompute all three expected costs. Which action now minimizes cost? State the additional evidence you would need to use these costs and probabilities in a real desk.

9. **Read a head's write.** The attention weights are $0.9$ and $0.1$, and the values projected onto a chosen direction are $-1$ and $8$. Compute both weighted contributions and their sum. Which token has the larger weight? Which contribution favors the positive direction?

10. **Integrate a saturated gradient.** For $F(u)=3u-2u^2$ from $u'=0$ to $u=1$, compute endpoint gradient times displacement, the integrated attribution, and the sum of the two values of the derivative at the midpoint fractions $0.25$ and $0.75$ divided by two.

11. **Fit a probe by hand.** Fit the centered one-feature probe to features $(-1,1)$ and labels $(0,1)$ with $\lambda=2$. Add the label mean back for test value $0.8$. Name one family that should be held out when evaluating transfer to new requests.

12. **Check two routes.** Suppose $g_{11}=4$, $g_{10}=3$, $g_{01}=3$ and $g_{00}=-1$. Compute the two individual knockout effects, the joint effect, and the interaction $g_{11}-g_{10}-g_{01}+g_{00}$.

13. **Pay for a larger code.** With $\mathbf h=(1,0.6)$ and the two codes in the section on sparse autoencoders, find the value of $\lambda$ where choosing one or two features gives the same objective. Change $\lambda$ to 0.4 and compute the two losses.

14. **Rescore a policy.** Take $\pi_{\mathrm{ref}}=(0.5,0.3,0.2)$, $r=(0,0,1)$ and $\beta=0.5$. Compute the three unnormalized masses and normalized probabilities. Check the result with `reward_tilt`. What happens if every reward increases by 2?

15. **Separate preference and policy scores.** A reward model scores two replies at 1.5 and 0.5. Compute the probability and loss for preferring the first, and repeat after adding 2 to both scores. In a DPO example the new and reference preference odds are 4 and 2. At $\beta=0.5$, compute the DPO margin and the resulting preference probability. Explain which choice each probability measures.

16. **Plan an audit.** An independent sample of 200 requests contains zero unauthorized writes. Under the common-rate assumption, compute the one-sided 95% upper bound. Give one reason a challenge set and an independent statistical audit can be useful for different purposes.
