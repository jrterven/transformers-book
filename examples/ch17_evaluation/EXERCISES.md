# Chapter 17 exercises

Designing Reproducible Evaluations

Exercises 1-5 use hand calculations, 6-9 use NumPy, and 10-12 design or audit comparisons. All twelve [worked solutions](../../solutions/ch17_evaluation/README.md) are available in every edition.

Use UPDATE as the positive class. Accuracy differences are B minus A. The binary Brier convention averages the squared error of the positive-class probability only. Bootstrap intervals are approximate percentile intervals; grouped draws preserve whole groups while the helper retains a per-message denominator.

1. **Choose an evaluation unit.** A system checks addresses over a conversation. In a short conversation it gets its only address check right. In a longer one it gets four of five checks right. Compute accuracy per check, the mean of the two conversation accuracies, and the fraction of conversations where every check is correct. Explain the weighting and the question answered by each number.

2. **Compare two kinds of error.** A dataset has six updates and fourteen lookups. Model A finds three updates and incorrectly flags one lookup. Model B finds five updates and incorrectly flags four lookups. Compute both confusion matrices, accuracy, precision, and recall. Let a missed update cost $c_{\mathrm{miss}}>0$ and a false alarm cost $c_{\mathrm{false}}>0$. Write each model's total cost and find which cost ratios favor model B.

3. **Make two reliability points.** Use scores $(0.1,0.2,0.4,0.6,0.8,0.9)$ and labels $(0,1,0,1,1,0)$. Put scores below 0.5 in one bin and the rest in another. Compute each point's coordinates. Which bin has a higher mean score than observed rate? Compute the binary Brier score and compare it with always predicting probability 0.5. Explain why the constant predictor can match the overall positive fraction without distinguishing the cases.

4. **Resample paired outcomes.** For five examples, the differences in correctness, B minus A, are $(0,1,1,-1,0)$. Compute the original mean. Compute the mean for the sampled one-based indices $(2,2,4,5,1)$. Explain what would be lost if A and B used separate sampled lists.

5. **Turn probabilities into decisions.** Assume an update probability $p=0.3$, zero cost for a correct decision, a missed-update cost of 4, and a false-alarm cost of 1. Compute the expected cost of each predicted label and derive the threshold for general positive costs. State a tie rule and explain which data may be used to tune the threshold.

6. **Check a split by group.** Create a NumPy array of group IDs $(1,1,2,3,3,4,5,5)$. Use `split_by_group` with counts $(3,1,1)$ and seed 7. Use each returned set of row indices to find its groups. Verify that no group appears in two splits. Compare each split's row count with its group count.

7. **Inspect an empty bin.** Call `reliability_bins` with labels $(0,0,1,1)$, scores $(0,0.1,0.9,1)$, and edges $(0,0.25,0.75,1)$. Which bins can appear on a reliability plot? Explain what would happen to the plot if an empty bin were drawn at zero.

8. **Change the resampling unit.** Use labels $(1,1,0,0,1,0)$, predictions A $(0,0,0,1,1,0)$, and predictions B $(1,1,0,1,0,0)$. Call `paired_accuracy_bootstrap` with `repeats=4000`, `confidence=0.95`, and `seed=5`. Repeat with group IDs $(0,0,1,1,2,2)$ passed as `groups`. Compare the point differences and the intervals. Say which procedure fits a dataset where each group comes from one independent customer, and why neither interval proves equivalence. Separately, customer U has one correctness difference $+1$ and customer V has differences $(-1,-1,0)$. Compute the per-message and equally weighted per-customer gaps. For the sampled customer list $(U,U)$, compute the numerator and denominator of the per-message gap.

9. **Audit a prediction join (optional).** Make six rows with distinct example IDs. Give them true labels and predictions, including at least one correct and one incorrect prediction. Shuffle the prediction records so the order-only accuracy changes. Use `binary_metrics` to compute accuracy first in the new row order, then after matching the IDs. Explain how you would detect a missing or duplicated ID.

10. **Separate length from evidence placement (optional).** Design a grid for finding an invoice number in a report. Choose three lengths and three relative evidence positions. Specify what to hold fixed within a row. Add a second experiment that holds the evidence-to-question distance constant. Describe where you would insert extra text as the report grows.

11. **Test one change at a time (optional).** Two released systems differ in their mixer, training data, and precision. One has lower response latency. Describe the claim that this comparison supports. Design a separate ablation to test the mixer and specify the quality, latency, and storage measures you would record.

12. **Design a reproducibility check (optional).** A colleague shares a claimed accuracy gain but gives only two aggregate scores. List the inputs and intermediate records you would ask for to check the split, align the predictions, and recompute a paired interval. Explain what extra evidence you need to assess variation across training runs.
