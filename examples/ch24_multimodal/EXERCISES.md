# Chapter 24 exercises

Vision-Language and Multimodal Models

Exercises 1-4 use hand calculations, 5-8 inspect the NumPy implementation,
and 9-11 design image-grounded evaluations. All eleven have
[complete solutions](../../solutions/ch24_multimodal/README.md).
Use `examples.ch24_multimodal.model`. Image rows and text columns keep
the same order in the scores and relevance labels.

1. **Normalize two modalities.** Use image rows $(3,0)$ and $(0,2)$ and text rows $(1,1)$ and $(0,2)$. Normalize every row and compute the logits with scale 2. Rank the texts for the first image.

2. **Read a row and a column.** Take logits $\left(\begin{smallmatrix}1&0\\2&3\end{smallmatrix}\right)$ with diagonal training pairs. Write the paired probability for the first image when comparing texts. Write the paired probability for the first text when comparing images. Explain why the same paired logit gives different probabilities.

3. **Count more than one valid caption.** A query has relevant captions C1 and C3. Its ranking is C2, C3, C4, C1. Find recall at ranks 1, 2, and 4. At those ranks, also record whether the retrieved list contains any relevant caption.

4. **Mix values before projection.** A query gives scores $(0,0)$ on two image values $(2,1)$ and $(0,3)$. Compute the attention weights and output. Then project the output using rows $(1,0,2)$ and $(0,1,-1)$. Give the shape of the projected row.

5. **Check row normalization.** Give the vectors from Exercise 1 to `contrastive_logits`. Multiply the first image row by a positive constant and run again. Then change the logit scale from 2 to 4. Compare the scores, rankings, and first-image probability of text 1 after row softmax. At scale 2, add a second copy of text 2 to the candidate list. Recompute that probability and explain why it changes even though the image and its original scores are unchanged.

6. **Transpose a contrastive batch.** Use the score matrix from Exercise 2. Call `symmetric_contrastive_loss` on it and on its transpose. The function returns the image-to-text loss, text-to-image loss, and mean in that order. Compare the three values and explain what the transpose exchanges.

7. **Build a rectangular retrieval task.** Use score rows $(2,1,0)$ and $(0,2,3)$ and Boolean relevance rows $(1,1,0)$ and $(0,0,1)$, with 1 meaning relevant. Call `bidirectional_recall_at_k` for $k=1$ and $k=2$. Compare the `per_query` and `mean` values in each direction.

8. **Change what a query reads (optional).** Use one query row $(0,\sqrt{2})$, key rows $(0,0)$ and $(0,\ln 2)$, and value rows $(3,0)$ and $(0,3)$. Use two-dimensional arrays with shapes $[1,2]$, $[2,2]$, and $[2,2]$. Give `image_feature_bridge` a projection with rows $(1,0,1)$ and $(0,1,-1)$. Unpack the context, weights, and projected row in that order. Repeat with both query coordinates negated and compare which value receives more weight.

9. **Keep related pairs together (optional).** A dataset has five captions per photo and several near-duplicate photos of the same scene. Design a split that tests retrieval on new scenes. Explain how to keep multiple valid captions in the evaluation and how to report recall for both retrieval directions.

10. **Test a spatial relation (optional).** Make an image pair in which a bus is to the left of a shelter in one image and to the right in the other. Keep the question fixed. Say how you would score both images as a pair. Compare with a baseline that receives only the question and explain what an unchanged answer tells you. For four pairs whose expected answers are (left, right), use predictions (left, right), (left, left), (right, left), and (right, right). Compute image accuracy, both-correct pair accuracy, and the fraction of pairs whose answers merely differ.

11. **Separate ambiguity from a wrong answer (optional).** Design a test with images that contain one bus, several buses of different colors, or no bus. Write a fixed question and decide which answers or requests for clarification to accept. Explain how you would report an unsupported color separately from a reasonable response to ambiguity.
