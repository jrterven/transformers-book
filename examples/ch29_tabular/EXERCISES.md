# Chapter 29 exercises

Learning from Tables

Exercises 1-4 use hand calculations, 5-8 use NumPy, and 9-11 design tabular
model comparisons. All eleven have [complete worked
solutions](../../solutions/ch29_tabular/README.md).
Import the functions from `examples.ch29_tabular.model`.
The 20-case example referenced in Exercise 8 has two groups of ten labels:
two positives in the first and eight in the second. Compare predictions 0.5
for everyone with 0.2 for the first group and 0.8 for the second.

1. **Decide when each feature exists.** A service predicts a cancellation at booking time. Classify the booking channel, the scheduled date, the cancellation fee actually charged, and the customer's earlier completed bookings as available or unavailable. Explain your assumption about the earlier bookings.

2. **Separate new and missing categories.** The training median is 5 and the learned levels are mail and web in that order. Encode rows $(\text{missing},\text{web})$, $(2,\text{missing})$, and $(2,\text{phone})$ in the chapter's six-column order.

3. **Compute a context-weighted prediction.** Give three context rows weights $(0.2,0.3,0.5)$ and labels $(1,0,1)$. Compute the weighted prediction, then state how changing the second label would affect it.

4. **Distinguish probabilities from decisions.** Two predictors give $(0.2,0.7)$ and $(0.4,0.6)$ for labels $(0,1)$. Compare their decisions at threshold 0.5 and their Brier scores. Explain what the probability score distinguishes. Repeat the score comparison if the labels are instead $(1,0)$. Does greater confidence always help?

5. **Keep a held-out row out of fitting.** Call `fit_row_encoder` on the four training rows. Pass the number and category columns as separate lists and use `None` for blanks. Encode the three held-out rows using the fitted median and levels. Append a fourth row $(12,\text{mobile})$ and encode again with the same fit. Check the column order and width.

6. **Test the effect of an unseen category.** Use the same fit to encode the numeric column `[6, 6, 6]` and the category column `['shop', 'kiosk', None]`. Compare the positions that change in the three output rows. Check whether their numerical entries agree. Encode $(6,\text{mobile})$ too. Can this representation distinguish two unseen channel names? Explain why an unknown flag alone does not teach a model the return rate of a new channel.

7. **Trace the influence of a context row.** Use `math.log` to make scores $(0,0,\ln 4)$. Pass them with labels $(1,1,0)$ to `context_probability`. Raise the last score by $\ln 2$ and compare the returned probability and weights. Explain why a larger score can lower the positive-class estimate.

8. **Score two confidence levels.** Use `brier_score` on labels $(0,1,1,0)$ with predictions $(0.2,0.7,0.8,0.1)$ and again with $(0.45,0.55,0.6,0.4)$. Compare both with accuracy at 0.5. Describe the additional evidence needed to discuss calibration. As a separate constructed example, reproduce the 20-case comparison in Figure 28.5; explain how both predictors can match observed frequencies yet have different Brier scores.

9. **Split for the intended customer.** A table has one to five orders for each of 500 customers. Design a split to test predictions for new customers. Then design one to test predictions on later orders from existing customers. State the grouping and time rules for each.

10. **Compare three methods fairly.** Choose a linear classifier, a tree ensemble, and a pretrained in-context model. Specify the training information each receives, where preprocessing and model choices are learned, and how you will compare performance as the training set shrinks.

11. **Investigate unreliable probabilities.** A model's 0.7--0.8 bin has an observed positive rate of 0.5 on the latest orders. Describe how you would check sample size, customer overlap, feature drift, and whether the gap also appears in an earlier validation period. Fit any recalibration on separate observations and evaluate it on a later untouched test set.
