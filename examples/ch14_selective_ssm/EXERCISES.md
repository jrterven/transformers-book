# Chapter 14 exercises

Selective State Spaces

Exercises 1-4 work through the arithmetic by hand. For Exercises 5-9, import NumPy as `np` and use `examples.ch14_selective_ssm.model`. Exercises 8-10 are optional investigations. All ten [worked solutions](../../solutions/ch14_selective_ssm/README.md) appear in each edition.

The scalar zero-order-hold update (Equation 14.2) is $x_t=a_tx_{t-1}+(1-a_t)u_t$, with $a_t=e^{-\Delta_t}$.

1. **Trace a selective state.** Use inputs $(4,10,2)$, retentions $(1/2,9/10,1/4)$, and $x_0=0$ in the scalar update of Equation 14.2. At each position, show the contribution of the old state and the new input separately. At position 2, what fraction of the distance from the old state to the input has been covered?

2. **Connect the step to the write.** Use $\Delta=\ln 2$ and $\Delta=\ln 4$ in the scalar zero-order-hold system. Find the retention and write for each step. Suppose the input differs from the starting state. Which step moves the state further toward the input? Find the control score $z$ that produces each step through $\Delta=\log(1+e^z)$, and check the corresponding sigmoid write.

3. **Respect time order.** The first scalar step maps $x$ to $0.8x+2$, and the second maps it to $0.5x-1$. Combine them into one affine map in time order. Check it on $x=3$. Then reverse the steps and compare the two maps. Finally append a third map $h(x)=0.25x+3$ and check that grouping the first two maps or the last two maps gives the same final transformation without reordering them.

4. **Follow one input to a read.** Take three positions with $a=(0.6,0.5,0.8)$. Use write columns $\mathbf{b}_1=(1,2)^\top$, $\mathbf{b}_2=(0,1)^\top$, $\mathbf{b}_3=(1,0)^\top$ and last read row $\mathbf{c}_3=(2,-1)$. Compute the three entries of row 3 of $M$. Label the transitions that each earlier write passes through. Explain why $a_1$ does not appear in this row's input coefficients. Does a zero coefficient for input 1 mean that its write vector was zero?

5. **Inspect a small step numerically.** Call `zoh_coefficients` on step sizes $10^{-4}$, $10^{-9}$, and $10^{-16}$. Compare its write coefficients with `1 - np.exp(-delta)` for the same NumPy array. Explain the difference in the last few digits or in the presence of a rounded zero.

6. **Give the coordinates different retentions.** Set $u=(2,0,1)$, make `a`, `b`, and `c` arrays of shape $[3,2]$. Give `a` rows $(0.25,0.75)$ and give `b` and `c` rows $(1,1)$. Run `selective_scan` from zero. Compare the two state coordinates after the second position. Then share the factor $0.25$ across both coordinates by giving `a` shape $[3,1]$ and compare the outputs.

7. **Move the chunk boundary.** Set $u=(1,2,0,3,1)$ and $a=(0.5,0.25,0.75,0.5,0.25)$. Use arrays of ones of shape $[5,1]$ for `b` and `c`. Call `ssd_dense` for the outputs and matrix, and `ssd_recurrent` for the outputs and state trace. Call `ssd_chunked` with chunk sizes 1, 2, 3, and 5. Compare each chunked output with the two references and each returned final state with the last recurrent state. Use `np.testing.assert_allclose`. For chunk size 3, separately calculate the last chunk's outputs from zero and the contribution from its incoming state, then verify their sum.

8. **Check causality within a chunk (optional).** Use the arrays from Exercise 7 and set the chunk size to three. Change the third input while keeping the coefficients fixed. Compare the first two outputs before and after the change, then identify a later output that responds to it. Explain how the local causal matrix enforces the boundary.

9. **Reset a state inside a chunk (optional).** Repeat Exercise 7 with chunk size 3. Set retention to zero and input to 4 at position 3, keeping every other entry fixed. Run with `x0=[2.0]` and again from zero. Compare the outputs. Which outputs depend on the initial state? Is the exact zero retention obtainable from a finite positive step in Equation 14.2 in exact arithmetic? Use the recurrence to explain why the current input at a reset position can still contribute.

10. **Design two memory tests (optional).** In one task, ask for the latest delivery location after several updates. In the other, ask for the exact first location after the same updates. Decide how to vary the number of intervening reports and where to insert irrelevant reports. Include sequences whose first and last locations differ. Describe how to measure accuracy and persistent state or cache bytes at each sequence length with the batch size and dtype held fixed. Explain how a state storing both the first and latest locations could solve these two tasks. Add a query about an unpredictable earlier update to probe whether the model retains more than those two facts.
