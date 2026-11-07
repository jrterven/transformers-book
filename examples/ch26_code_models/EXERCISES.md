# Chapter 26 exercises

Models for Code

Exercises 1-4 use hand calculations, 5-8 use known Python callables, and
9-11 design repository evaluations. All eleven have [complete worked
solutions](../../solutions/ch26_code_models/README.md).
The functions are in `examples.ch26_code_models.model`; the runnable module
exports `CANDIDATES` in the chapter figure's order.

1. **Check the boundaries.** For each pair $[0,2)$ and $[1,4)$, $[1,4)$ and $[4,7)$, and $[2,2)$ and $[0,5)$, list the shared integers. Explain why the right endpoint is excluded.

2. **Reorder an edit.** Write a prefix and suffix around a missing function body. Show which pieces the model sees before generating the body in the prefix--suffix--middle format. Identify a clue in the suffix that helps specify the body.

3. **Distinguish two expressions.** Consider the candidates $\ell\le r$ and $\ell<r$. Supply one case where both return true, one where both return false, and one where they disagree. Explain which case the smoke suite needs to add to detect the boundary error.

4. **Count successful pairs.** Imagine five candidates with exactly two passing a task. Count size-two subsets, then count the subsets that contain at least one pass. Compute the corresponding $\mathrm{pass@}2$.

5. **Inspect the evaluation record.** Run Listing 26.5. In `benchmark.candidates`, find `touching_counts` and `needs_two`. Read their `edges.per_case` records and identify a different failed case for each. Explain each result from the corresponding comparison in the function. Across all candidates, compute the fraction of passing individual checks in the complete six-case suite and the fraction of candidates passing every check. Explain why these fractions differ.

6. **Add one boundary case.** Create a new `Case(name, a, b, expected)` with two negative-start intervals and exactly one shared integer. State its expected result before passing a one-element case list to `score_candidate` for each of the four candidates. Check which outcomes agree with the contract.

7. **Check the metric.** Call `pass_at_k(5, 2, 2)`. Compare its value with Exercise 4. Repeat with $k=1$ and $k=5$. Explain the meaning of each setting. Then use two tasks with $(n,c)=(4,2)$ and $(4,0)$ to compute mean $\mathrm{pass@}2$. Compare it with pooling the candidates as $(8,2)$ and explain which calculation matches the task definition.

8. **Separate selection from coverage.** Three candidates have 12, 18, and 7 lines, respectively. Only the first two pass. Score the rule that always picks the shortest candidate, then compute $\mathrm{pass@}2$ for the pool. Explain which quantity would predict what the user sees under this rule.

9. **Design a repository-context comparison.** Choose a function whose behavior depends on a helper in another file. Specify a local-only condition and a retrieved-context condition. Keep the model and generation budget fixed. Decide which held-out cases could show an improvement from the retrieved helper.

10. **Check for regressions.** An issue says adjacent bookings are incorrectly reported as overlapping. Describe one test that must change from failing to passing and two that should remain passing. Explain why a patch that always returns `False` can fix the reported case and still fail the complete task. State what information the issue report gives the model and what you keep for evaluation.

11. **Audit a benchmark split.** You have tasks derived from public issues, patches, and tests. Propose a split by repository and date. Explain how you would detect near-duplicate tasks, prevent reference-patch leakage, and report uncertainty about pretraining overlap.
