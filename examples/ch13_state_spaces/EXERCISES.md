# Chapter 13 exercises

Structured State Spaces and Long Convolutions

Exercises 1-4 trace a state and its filter by hand. For Exercises 5-9, import NumPy as `np` and use `examples.ch13_state_spaces.model`. Exercises 9-11 are optional extensions. All eleven [worked solutions](../../solutions/ch13_state_spaces/README.md) appear in each edition.

1. **Follow one state.** Let $x_t=0.25x_{t-1}+u_t$ with $x_0=0$. Compute the states for inputs $(4,0,0,2)$. At the last position, separate the contribution of the first input from that of the last input. Then compare the final states for two shorter histories, $(4,0)$ and $(0,1)$. Can this state alone distinguish them after the second input?

2. **Read by lag.** For the system in Exercise 1, use $y_t=x_t$. Write the first four filter coefficients in order from lag zero. Use them to compute $y_3$ and $y_4$ as weighted sums of the inputs. Compare these outputs with the recurrence.

3. **Carry two features.** Use $\overline{A}=\operatorname{diag}(1/2,3/4)$, $\overline{B}=(1,1)^\top$, and $C=(1,-1)$. Start from zero and process inputs $(4,0,0)$. Write each two-coordinate state and each scalar output. Explain how the output can change after the input becomes zero. Does a zero first output mean that the state is empty?

4. **Choose a step size.** For the scalar continuous system with $A=-1$ and $B=1$, use the bilinear rule with $\Delta=1/2$ and with $\Delta=1$. Compute $\overline{A}$ and $\overline{B}$ for each. Which update retains more of a nonzero state when the new input is zero?

5. **Compare three calculations.** With the parameters and input sequence of Exercise 3, store $\overline{B}$ and $C$ as one-dimensional arrays. Call `ssm_scan`, which returns the outputs and the states. Use `ssm_kernel` with a filter length equal to the input length. Apply both `causal_convolution` and `fft_causal_convolution`. Compare all three output arrays with `np.testing.assert_allclose`, using `rtol=0` and `atol=1e-12`.

6. **Keep the tail in its place.** Use input $(3,0,1,0,2)$ and filter $(1,-1,1)$. Find the minimum transform length for linear convolution of the full arrays. Compute the causal output with the provided FFT function. Then compute a circular convolution by passing `n=5` to both `rfft` calls and to `irfft`. Explain which output positions receive a folded contribution. Increase only the last input by 10. Check whether earlier outputs change for the correctly padded calculation and for the length-five circular calculation.

7. **Resume from a saved state.** Use `a_bar=[[0.75]]` and `b_bar=c=[1.0]`. Run `ssm_scan` on $(1,4,0,2,1)$. Run it again on the first two inputs, then on the last three with `x0=states[-1]` from the first chunk. Compare the joined output with the single run. Repeat with the second chunk starting at zero and explain the difference.

8. **Move a zero gate.** Call `hyena_stage` with inputs $(1,2,1,1)$, input gate $(1,0,1,1)$, an output gate of ones, and filter $(1,1/2,1/4)$. Repeat with the input gate of ones and output gate $(1,0,1,1)$. Compare the gated input, filtered values, and final output. Which later positions are affected by zeroing the input at position 2?

9. **Account for the initial state (optional).** Use $\overline{A}=\operatorname{diag}(1/2,1/4)$, $\overline{B}=(1,1)^\top$, $C=(2,1)$, $\mathbf{x}_0=(1,-1)^\top$, and input $(0,3,1,0)$. Compute the output using the recurrence. In a second calculation, add $C\overline{A}^t\mathbf{x}_0$ to the zero-state convolution, starting at $t=1$. Use `np.linalg.matrix_power` to compute the powers. Compare the results.

10. **Count streaming storage (optional).** Compare a linear state space layer with 16 state coordinates per feature and a direct streaming convolution with a filter of length 1024. Count the numbers carried from one step to the next for one feature. Exclude model parameters and temporary work. Repeat when the filter length doubles while the state width stays fixed. Does the smaller state count establish that the SSM can reproduce any arbitrary filter of that length? Explain.

11. **Check a changing rule (optional).** Consider $x_t=a_t x_{t-1}+u_t$ and $y_t=x_t$ with $x_0=0$. Use $a_2=1/2$ and $a_3=3/4$. Compare the coefficient of $u_1$ in $y_2$ with the coefficient of $u_2$ in $y_3$. Both use lag one. Explain what this implies about using one fixed filter for the sequence. Also find the coefficient of $u_1$ in $y_3$. If the factors $a_t$ are prescribed independently of the inputs, is the map still linear in the inputs?
