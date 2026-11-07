# Chapter 18 exercises

Language Understanding

Exercises 1-5 use hand calculations, 6-9 use NumPy, and 10-12 address evaluation in a new setting. All twelve [worked solutions](../../solutions/ch18_understanding/README.md) are available in every edition.

Import code helpers from `examples.ch18_understanding.model`. Use IOB2 tags, zero-based word indices and exclusive span ends. Keep sentence identity when aggregating span counts. The supplied encoder features and token splits are teaching fixtures.

1. **Find what word counts lose.** Compare *Ari called Sam* with *Sam called Ari*. Write a word-count vector for each using the same vocabulary order. Explain what a contextual encoder needs to distinguish who made the call, and why bidirectional attention without any position signal is insufficient.

2. **Pool only the valid rows.** Use feature rows $(2,0)$, $(1,3)$, $(0,1)$, and $(6,6)$ with valid mask $(1,1,1,0)$. Compute the pooled vector. Compare it with the unmasked mean. Give a condition under which adding a padding row would leave the unmasked mean unchanged.

3. **Assign entity tags.** In *Sam left Lake City*, mark *Sam* as a person and *Lake City* as a location using IOB2. Now suppose a tokenizer splits *Lake* into two pieces. Draw the first-piece supervised labels, including a beginning special token with no word ID. Explain the difference between an `O` target and an ignored target. Would the second piece need to be hidden from attention?

4. **Compare tag and span accuracy.** Use gold tags `B-ORG, I-ORG, O, B-PER, O`. The prediction is `B-ORG, I-ORG, O, B-LOC, O`. Write the gold and predicted typed spans. Compute word accuracy, exact span precision and recall, and the true-positive, false-positive, and false-negative entity counts. Explain what matching only the start and end would miss.

5. **Keep sentence identity when scoring.** Sentence A has one gold person at $(0,1,\mathrm{PER})$ and no predicted entities. Sentence B has no gold entities and one predicted person at those same offsets. Compute dataset precision, recall, and $F_1$. What would go wrong if you removed the sentence IDs before taking the set intersection? In a separate dataset, sentence C has counts $(M,P,G)=(1,1,1)$ and sentence D has $(1,3,3)$. Compare micro $F_1$ with the mean of the two sentence $F_1$ scores.

6. **Change the padding rows.** Call `sentence_logits` with the four rows from Exercise 2, a Boolean mask, `np.eye(2)` as the weights, and `np.zeros(2)` as the bias. Replace the last feature row with $(20,-10)$ and call the function again. Compare the pooled vectors and logits. Then mark the last row valid and check what changes. Explain why this proves invariance for the supplied features but not for an encoder run with an incorrect padding attention mask.

7. **Align a split word.** Use word IDs $(-1,0,0,1,2,2,-1)$ and word labels $(4,0,2)$. Call `first_piece_labels`. Use `first_piece_predictions` with the same word IDs and token predictions $(9,4,8,0,2,7,9)$. Which token predictions are ignored under this policy?

8. **Choose a repair policy.** Use tags `B-LOC, I-LOC, I-PER, I-PER, O`. Call `bio_spans` with its default policy and record the exception. Then call it with `repair=True`. Explain what the second call does with the preceding location span.

9. **Test a boundary error (optional).** Use gold tags `O, B-LOC, I-LOC, O, B-PER`. Compare two predictions that are identical to the gold sequence except that one changes the third tag to `O` and the other changes the fifth tag to `B-ORG`. Make a third prediction that matches gold except for `B-ORG` on the first word. Use `span_metrics` for each case. Compare the matched entities, precision, and recall.

10. **Change the label mix (optional).** Two offices each receive 200 messages. The first receives 60 updates and the second receives 20. A classifier finds 90\% of updates and falsely flags 5\% of lookups in both offices. Compute the expected counts and precision for each office. Name one additional change you would check on the second office's data.

11. **Plan an independent split (optional).** Design a dataset with repeated templates from several offices and some customers who contact more than one office. Specify which group you would hold out to test new customers and which to test a new office. Describe how you would detect leakage through paraphrases.

12. **Design an entity error study (optional).** Compare two taggers on place names seen in training and names absent from training. Define how to identify each slice, what counts as a boundary or type error, and how to report each slice's support and score. Explain how to assign an incomplete predicted name to a slice. Include a check for names broken into multiple subwords.
