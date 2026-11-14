# Chapter 21 exercises

Search, Retrieval, and Grounded Assistants

Exercises 1-5 use hand calculations, 6-9 investigate the NumPy helpers,
and 9-12 are optional extensions in source auditing and evaluation design.
All twelve have [complete worked solutions](../../solutions/ch21_retrieval/README.md)
in the notebook and both PDF editions.

Import helpers from `examples.ch21_retrieval.model`. The small collection is:

| ID | Text |
| --- | --- |
| P1 | Renew a parking permit at the mobility desk. |
| P2 | A parking permit expires on June 30. |
| P3 | The recycling office opens on Tuesday. |

For Exercise 1, use positive IDF `ln(1 + (N - df + 0.5)/(df + 0.5))`.
For Exercise 2, the passage vectors in P1/P2/P3 order are `(0.8,0.6)`,
`(0.6,0.8)`, `(0,1)`. Packing visits every ID in rank order, keeping
a whole passage if it fits, and continues after a skipped passage.
RAG-Sequence averages passage-conditioned path probabilities;
RAG-Token averages the conditional probabilities at each token position
before multiplying over the chosen prefix. Both use the original retrieved set.

1. **Compare two query words.** In a collection of ten passages, one term occurs in two passages and another in eight. Compute the two IDFs in Equation 20.1. If their frequencies and length adjustments are equal in a candidate, explain which term contributes more.

2. **Change the query direction.** Use the three passage vectors from Figure 20.1 and query $(0,1)$. Compute the dot products and rank the passages. Then give a condition on two positive query coordinates under which P1 would rank ahead of P2.

3. **Fit a short context.** Four passages are ranked P1, P2, P3, P4. They cost 9, 5, 4, and 2 tokens. With a budget of 11, apply the whole-passage packing rule from this chapter. Identify a budget at which P2 is included and P3 is excluded. Explain how the ranking and the passage lengths both matter.

4. **Count missing evidence.** A question needs passages P2 and P4. The search ranking is $(\mathrm{P3},\mathrm{P4},\mathrm{P1},\mathrm{P2})$. Compute recall at ranks one, two, and four. For each rank, record whether any evidence was found and whether both required passages were found.

5. **Mix before or after multiplying.** Two retrieved passages have weights 0.6 and 0.4. For a fixed candidate $(y_1,y_2)$, passage A gives probabilities 0.8 and 0.3, while passage B gives 0.2 and 0.9. The second probabilities are conditional on $y_1$. Compute the candidate prefix probability under RAG-Sequence and RAG-Token. Explain why the results differ and whether either calculation requires a new retrieval for $y_2$.

6. **Turn off length adjustment.** Pass the three texts from Table 20.1 in ID order to `bm25_scores` for query *parking permit*. Use `rank_passages` with the IDs in the same order. Run once with the defaults and again with `b=0`. Compare the scores for P1 and P2 and explain the order when scores tie.

7. **Rerank only what was found.** Use the candidate IDs P2 and P3. Pass `rerank_candidates` a dictionary with scores 0.2 for P2 and 0.7 for P3. Compare the result with one where the candidate set includes P1 and its score 0.9. For each case use `evidence_recall` with relevant IDs `('P1',)` and $k=1$.

8. **Keep the passage and ID together.** Build an ID-to-text dictionary from Table 20.1 and use ranking P2, P1, P3. Call `pack_context` with budgets 7 and 14. Then move P1 to the front and repeat with budget 14. Pass the returned contexts to `literal_evidence_hits` for *mobility desk*. Explain how packing can remove evidence already retrieved.

9. **Probe a literal match (optional).** Create two (ID, text) pairs. Let P4 say *Express renewals are available* and P5 say *Express renewals are no longer available*. Use `literal_evidence_hits` for *express renewals*. Compare the matches with what each passage says about whether express renewals are available.

10. **Choose a passage boundary (optional).** Write a two-sentence notice with an instruction and an exception. Compare splitting the notice into two passages with keeping both sentences together. State how to test retrieval and whether the answer preserves the exception.

11. **Design an abstention test (optional).** Write a question that is answerable from P2, one that is topically related to P2 but unanswerable from the three notices, and one on an unrelated topic. Define what the system should return for each and how you would score appropriate answers and abstentions.

12. **Audit two versions (optional).** Add an old notice that directs permit renewals to the records counter. Decide what metadata the collection must record. Design an evaluation that tests whether the system retrieves the current notice, cites its ID, and distinguishes it from an older notice that contains the same query words.
