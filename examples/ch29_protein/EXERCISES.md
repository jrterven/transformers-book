# Chapter 29 exercises

Scientific Sequences and Biological Structure

Exercises 1-4 use hand calculations, 5-8 use NumPy, and 9-11 design scientific
evaluations. All eleven have [complete worked
solutions](../../solutions/ch29_protein/README.md).
Import the functions from `examples.ch29_protein.model`.

Fixtures referenced by the exercises:

- Alignment rows: ACDG, ACEG, ASDA, ASEA; highlighted columns: 2 and 4 (one-based).
- Pooling rows: BOS=(9,9), A=(1,0), C=(1,2), EOS=(-9,-9), PAD=(0,0).
- Four-point geometry: (0,0,0), (3,0,0), (0,4,0), (8,0,0), in chain order.
- Constructed confidence: domain A positions 1-3, domain B positions 4-6;
  pLDDT=(94,96,95,95,97,94); PAE diagonal 0, other within-domain entries 2,
  rows A/columns B 18, rows B/columns A 12 angstroms. Rows identify the error
  position; columns identify the alignment anchor. These are teaching values.

1. **Read aligned columns.** In Figure 29.1, count the observed letter pairs at the two highlighted positions. Describe one reason the association might appear even if the residues are far apart in space.

2. **Preserve some order.** Compare `CAAC` and `CCAA`. Write their composition entries for A and C. Then list the ordered adjacent pairs and compute their normalized counts.

3. **Pool only residues.** Use vectors $(10,10)$, $(2,0)$, $(0,4)$, and $(-10,-10)$ for a beginning marker, two residues, and an ending marker. Compute the mean over the residues and the mean over all four positions. Explain why the residue mask matters.

4. **Read a geometric threshold.** Place three residues at $(0,0,0)$, $(2,0,0)$, and $(0,3,0)$. Use an inclusive distance threshold of 3 and minimum index separation two. Use one-based indices to identify the accepted unordered pairs and their matrix entries.

5. **Verify the pair representation.** Use `residue_composition` and `dipeptide_frequencies` on the sequences in Exercise 2. Check the sum of each feature vector or matrix. Repeat with a sequence containing just A and explain the difference between the two sums. Then compare `AACAC` and `ACAAC`. Can dipeptide frequencies always recover sequence order?

6. **Change only padding.** Call `mean_pool_residues` on the five rows from Figure 29.2, with a mask of `True` for A and C and `False` for all other rows. Change the padding row to $(7,9)$ and run the function with the same mask. Compare the results.

7. **Translate the coordinates.** Convert the four-point coordinates to a NumPy array. Call `contact_map` with threshold 5 and minimum index separation 2. Add $(10,-3,2)$ to every point and recompute with the same settings. Explain why a shared translation can or cannot change any entry. Also rotate every point by $(x,y,z)\mapsto(-y,x,z)$ and check the map. Explain what changes if the threshold is moved just below 5.

8. **Measure two classes.** Use true labels `['A', 'A', 'A', 'B']` and predicted labels `['A', 'A', 'B', 'B']`. In Python, count true positives, false positives, and false negatives for each class in the four-protein example. Compute each F1 and macro-F1. Keep the true labels fixed and repeat after changing the last prediction to A.

9. **Split by relatedness.** You have sequences grouped by protein family and labeled by location. Design a split to test generalization to families absent from training. Explain how you would handle a class represented by only one family and report what the test can measure.

10. **Test what an embedding adds.** Design a comparison of composition, dipeptides, and frozen residue vectors. Keep the labeled data split and classifier family fixed. Specify the pooling rule, what to record about pretraining overlap, and how you would check whether sequence length explains a score difference.

11. **Assess a structural claim.** A predicted protein has two locally confident domains and uncertain relative orientation. A proposed binding site spans both domains. Use the constructed confidence plot to explain the relevant local and cross-domain information. State whether PAE should be read as a distance or as a predicted error, and describe what experimental evidence could test the binding claim.
