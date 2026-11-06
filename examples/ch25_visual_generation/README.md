# Chapter 25. Image and Video Generation

Run the offline examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch25_visual_generation.run_examples --output output/ch25.json
python -m unittest discover -s tests -p test_visual_generation.py -v
```

## Mix a clean sample and noise

`add_noise` takes a clean sample, a supplied noise array of the same
shape, and a scalar cumulative signal coefficient `alpha_bar`.
The endpoints zero and one give the supplied noise and the clean
sample, respectively. The example uses two-by-two arrays to make
every multiplication visible.

## Guide an epsilon prediction

`guided_noise` takes unconditional and conditional predictions
with the same shape and a scalar guidance scale. Scale zero gives
the unconditional prediction. Scale one gives the conditional
prediction. Larger positive values extend the displacement from
the unconditional prediction toward the conditional prediction.

## Take a deterministic DDIM step

`ddim_step` takes a noisy sample, its predicted epsilon, the current
`alpha_bar_t`, and the next `alpha_bar_next`. The schedule follows
`0 < alpha_bar_t <= alpha_bar_next <= 1`. The function returns
`(estimated_clean, next_noisy)`. It first reconstructs the clean
estimate and then combines that estimate and the predicted epsilon
at the next noise level. The deterministic update uses no new
random draw.

The example follows `0.36 → 0.64 → 1.0` with the exact epsilon and
with one epsilon entry shifted by 0.25. It holds the supplied
prediction fixed across the two steps in each run. The difference
in the final arrays isolates how a prediction error enters this
calculation. In model sampling, a denoiser can produce a new
prediction at each step.
