# Chapter 26 exercises

Image and Video Generation

Exercises 1-4 use hand calculations; 5-8 use the canonical NumPy helpers;
9-11 design evaluations. All eleven have [complete worked solutions](../../solutions/ch26_visual_generation/README.md).
The toy arrays need no model download. Evaluation-design exercises do not
claim measured results for a pretrained generator.

1. **Mix signal and noise.** Use scalar clean value $2$, noise value $-1$, and $\bar\alpha_t=0.64$. Compute the noisy value. Repeat with $\bar\alpha_t=0.36$ and compare the signal and noise weights.

2. **Recover a clean estimate.** For the first mixture in Exercise 1, use predicted noise $-0.5$. Compute $\widehat{x}_0$ and the next state at $\bar\alpha_s=0.81$. Compare both outputs with the exact-noise case. Separate the change in the next state into its clean-estimate and predicted-noise contributions.

3. **Change a guidance scale.** Let the unconditional prediction be $(0,1)$ and the conditional prediction be $(2,0)$. Compute the guided vectors for $g=0$, $1$, and $1.5$. Describe how the third point relates to the first two.

4. **Count video attention pairs.** A clip has five frames and six patches per frame. Count the ordered pairs including self-pairs for joint attention, per-frame spatial attention, and per-site temporal attention. Explain why the count alone cannot predict runtime.

5. **Check the endpoints.** Call `add_noise` with two different $2\times2$ arrays at $\bar\alpha=0$ and $\bar\alpha=1$. Check the values and the output shape at both endpoints. Then try `ddim_step` with current level zero and next level 0.64. Explain why this endpoint is valid for mixing but invalid for that sampler formula.

6. **Read both DDIM outputs.** Use the noisy grid at $\bar\alpha_t=0.36$ from Figure 25.1 and the supplied exact noise. Call `ddim_step` with next level 0.64. Inspect both outputs. Pass the next sample and the same noise to a second call with current level 0.64 and next level 1.

7. **Change one prediction.** Keep the same initial noisy grid as in Exercise 6. Make a copy of the noise, add 0.1 to its lower-left entry once, and pass the altered prediction to both sampler calls. Compare all four entries with the exact-prediction run. Explain the behavior using Equation (25.3).

8. **Sweep guidance.** Use `guided_noise` on the two vectors in Exercise 3 at five equally spaced scales including zero and two. Compare the difference between successive guided vectors. Connect it to the conditional minus unconditional prediction.

9. **Compare two sampling settings.** Design a comparison between 10 and 30 steps for one pretrained generator. Keep the checkpoint, scheduler family, prompts, initial noise tensors, and guidance fixed. Specify the timing boundaries and count denoiser evaluations, including both predictions for classifier-free guidance when they are used. Decide which aspects of output quality to measure separately.

10. **Check motion as well as appearance.** Design a prompt set with an object that turns, a moving camera viewing a stationary object, and a stationary camera viewing a static scene. Write one check for each prompt that requires more than one frame. Decide how to record cases where appearance is correct and motion is wrong.

11. **Separate variety from prompt following.** Compare two guidance scales across multiple noise draws. For each prompt, use the same draws at both scales. Describe how to measure prompt adherence and how to detect repeated compositions. For one prompt, suppose eight samples occupy four composition categories with counts $(4,2,1,1)$ at one scale and $(3,2,2,1)$ at the other. Compute the fraction of unordered sample pairs in the same category for each scale. Explain what this reveals, what it does not reveal, and what changes if you save only the strongest sample for each prompt.
