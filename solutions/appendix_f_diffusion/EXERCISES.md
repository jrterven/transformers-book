# Appendix F | Exercises

Diffusion Foundations

Exercises 1-4 establish the basics, 5-6 apply them, and 7-8 diagnose errors. Complete explanations and executable checks accompany every exercise in this directory.

## 1. Read a noise schedule

For beta values 0.1 and 0.2, compute alpha values and cumulative alpha at step 2. Which quantities multiply the clean sample and the noise?

## 2. Convert a velocity prediction

At $a=0.8$, $b=0.6$, the noisy value is 0.5 and predicted velocity is $-0.25$. Convert to clean and noise estimates. What does an epsilon sampler expect?

## 3. Take a deterministic step

Call `ddim_step` for noisy value 0.3, predicted noise $-0.5$, cumulative alpha 0.64 and next cumulative alpha 0.81. Identify both outputs and the endpoint when next cumulative alpha is 1.

## 4. Keep frame and noise axes distinct

A video has six frames, height 32, width 48, and three pixel channels. Give noise shape in frame/height/width/channel order. If conditional and unconditional predictions differ, which guidance scale returns the conditional one? What shape changes?

## 5. Add independent variances

For two beta values 0.2, derive the two noise coefficients and combined variance. What if both increments reuse the same Gaussian? Also find noisy variance when the clean coordinate has variance four, independent of the noise.

## 6. Test the rotation including endpoints

Choose fixed clean and noise arrays. For $a=0,0.8,1$ and $b=\sqrt{1-a^2}$, form noisy values and velocities, then recover both inputs. Which endpoint prevents epsilon-to-clean division?

## 7. Measure error amplification

Hold a clean scalar and noise draw fixed and add noise-prediction error 0.02. Compare clean-estimate errors at $a=0.8$ and $a=0.1$ with $-(b/a)\delta$. Why is this not a denoiser benchmark?

## 8. Check guidance and conversion order

Use noisy value 0.5, $a=0.8$, $b=0.6$, unconditional velocity $-1$, conditional velocity 0.5, and guidance 2. Compare guidance before/after conversion to noise. Then clip the guided velocity before conversion versus clipping final guided noise, both to $[-1,1]$.
