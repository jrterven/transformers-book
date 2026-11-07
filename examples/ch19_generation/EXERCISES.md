# Chapter 19 exercises

Translation and Summarization

Exercises 1-5 use hand calculations, 6-9 investigate the NumPy helpers,
and 9-12 are optional extensions on search and source-based evaluation.
All twelve [worked solutions](../../solutions/ch19_generation/README.md)
appear in both PDF editions and the executed notebook.

Import helpers from `examples.ch19_generation.model`. A generated prefix
excludes BOS and includes EOS when finished. Probability products include
EOS; the output cap counts it as one generated token. Overlap helpers take
token lists rather than raw strings.

1. **Decide what must be preserved.** A report says the last ferry left at 19:10, carried thirty passengers, and was delayed by a gate inspection. Write which facts a translation must preserve. Choose two details for a one-sentence summary for a passenger who wants to know why the ferry was late. Explain your selection.

2. **Align two target lengths.** Take target token rows $(5,7,6)$ and $(8,9)$. With `PAD`=0, `BOS`=1, and `EOS`=2, write the two decoder-input rows and their aligned label rows. Mark each label position that contributes to the loss.

3. **Compare complete paths.** The first token is A with probability 0.7 or B with probability 0.3. After A, the probabilities of `EOS` and C are 0.6 and 0.4. After B, they are 0.9 and 0.1. Suppose C is always followed by `EOS`. Compare the completed paths. Which one would greedy decoding choose, and which has highest total probability?

4. **Count repeated and ordered matches.** Use candidate tokens *red red ferry departs* and reference tokens *red ferry departs at noon*. Find clipped unigram precision and one longest common subsequence. Explain why the repeated word does not earn a second match.

5. **Aggregate counts before dividing.** Two examples have clipped unigram match counts 3 and 4, with candidate lengths 4 and 10. Find the corpus unigram precision and the mean of the two sentence precisions. Explain what receives equal weight in each calculation. Is either number, by itself, a complete BLEU score?

6. **Inspect a padded batch.** Use source rows $(3,4)$ and $(5,6,7,8,9)$ with the targets from Exercise 2. Pass a list of the two source--target pairs to `collate_pairs` and check its three outputs. Compare source and target widths, then count the valid labels.

7. **Stop at a length cap.** Write a callback for `beam_search` with A=3 and `EOS`=2. Return a dictionary from token IDs to probabilities. From an empty prefix, give A probability 1. From the singleton prefix `(3,)`, give `EOS` probability 1. Call the search with beam width one and maximum steps one and then two. Compare the tokens and completion flags.

8. **Compare overlap measures.** Take the tokens from Exercise 4. Use `clipped_ngram_precision` for $n=1$ and $n=2$. Then call `rouge_l`. Swap the candidate and reference in both functions. Explain which counts or denominators can change.

9. **Keep a useful prefix (optional).** Use A=3, B=4, C=5, D=6, and `EOS`=2. Let the root probabilities be 0.65 for A and 0.35 for B. After A let C have probability 0.7 and `EOS` 0.3. After B give `EOS` probability 1. After $(\mathrm{A},\mathrm{C})$ give `EOS` probability 0.6 and D probability 0.4. Compare beam widths one and two with maximum steps two and then three. Report the completion flags and explain what changes at the third step.

10. **Check a summary claim (optional).** A notice says the ferry was late and the north gate was closed. A candidate says the ferry was late *because* the north gate was closed. Break the candidate into claims and identify the parts that the notice supports. Explain how an overlap score might respond.

11. **Trace a missing fact (optional).** Design a summary task that explicitly requests a specific fact. Keep the instruction visible and place the answer-bearing sentence before or after the input limit. Specify how to check the actual input supplied to the encoder. Explain how to separate an omission caused by preprocessing from one caused by generation.

12. **Plan a comparison (optional).** Compare a compact pretrained encoder--decoder before and after adaptation on parallel text or document summaries. Specify the split groups, validation criteria, generation settings, overlap metric, and a source-based check. Decide how you would report unsupported additions separately from coverage.
