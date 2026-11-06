# Chapter 7: GPT, BERT, and T5 - Exercises

Exercises 1-7 use short traces, diagrams, and calculations; 8-11 inspect the learning problems in code; 12-14 are optional investigations. Use the [chapter guide](README.md) for the teaching vocabulary and APIs. In numerical probes, keep weights fixed, use evaluation mode, and state the comparison tolerance.

For span corruption, $N$ is the original word count, $R$ the number of removed words, and $K$ the number of separated spans. This chapter uses one new terminal sentinel and an EOS marker.

## 1. Move the answer one place ahead

Write the input and label arrays for *the red car stopped*. For each output position, list the words it may read under causal attention. Explain why allowing the diagonal does not reveal the next-token answer. Repeat with a five-token sentence of your own.

## 2. Draw the information paths

Draw a four-position causal self-attention grid and a four-position bidirectional grid. Count the permitted cells in each. Then draw cross-attention with three decoder queries and four encoder keys. Label the axes and explain why the cross-attention grid can be rectangular. Describe how one padded encoder key changes its permitted column.

## 3. Separate selection from replacement

For 200 eligible tokens, calculate the expected selected count under BERT's 15% rate. Among the selected positions, calculate the expected counts for mask, random, and unchanged treatments. What is the expected number of labels contributing to the masked-token loss? Explain why a selected token that remains unchanged still has a label, and why sampled treatment counts need not equal their expectations.

## 4. Keep three masks separate

For original text *the red car stopped*, supply `[the, [MASK], car, stopped, PAD]` and select only *red* as a label. Write the key-validity row and the complete label row using `IGNORE`. Can the selected query read `[MASK]`? Can it read `PAD`? If the correct-token probability is $0.8$, compute the mean loss and the incorrect values obtained by dividing the loss sum by four or five. Explain what each of the three decisions changes.

## 5. Put the fragments back in order

Number the words in *the red car stopped near the old bridge today* from zero. Apply spans `(1, 3)` and `(6, 8)`. Write the corrupted source, reconstruction target, and shifted decoder input, including all markers used in this chapter's convention. Count the tokens in each. Explain why *stopped near the* is not repeated in the target. Then use just the span `(1, 4)`. Calculate source and target lengths from $N$, $R$, and $K$ before checking with `span_corrupt`.

## 6. Check a loss before training

Suppose all $V$ vocabulary logits are zero. Calculate the mean loss and loss sum for the three-position GPT example, the one-selected-position BERT example, and the eight-position T5 target. Set $V=17$, the size of the teaching vocabulary, for a numerical answer. What changes if two ignored padding labels are appended to each target? Distinguish this arithmetic check from a comparison of learned model quality.

## 7. Choose what changes

A pretrained token model is reused in three ways: supply a new prompt with fixed weights; freeze its stack and train a new two-class head; fine-tune the stack and that head. State which parameters change in each case. For a causal input *the red car stopped*, compare pooling the first and last valid positions. Which can read *red*? Calculate the two-class example in $p(c\mid x)=\operatorname{softmax}(hW+b)_c$, with row vector $h=(1,2)$, $W=\begin{bmatrix}0.5&-0.5\\1&0\end{bmatrix}$, and zero bias, and explain why calling `eval()` does not by itself freeze parameters.

## 8. Follow all three BERT treatments

Start with *the red car stopped*, select Python index 1, and create one `torch.Generator().manual_seed(19)`. Reuse it across calls to `bert_corrupt` so its random state advances. Pass `return_trace=True` to receive the sampled `draw` and treatment for each call without changing the helper. Find an example of each treatment. Compare the original IDs, corrupted IDs, and labels to verify that selected labels keep their original IDs and unselected labels equal `IGNORE`. Explain how the trace distinguishes keeping the original word from drawing that same word as a random replacement.

## 9. Find where the direct loss acts

Construct random logits with shape $[1,4,V]$, enable gradient recording, and use labels with only Python index 1 (the second position) selected. Call `selected_loss` and backpropagate. Inspect all four logit-gradient rows. Explain why zero direct loss gradient at an unselected output does not imply that its input embedding receives no gradient in a bidirectional network.

## 10. Test a future-token boundary

Run `visibility_probe`. Then repeat the check with *the blue car stopped* and *the blue car moved*. Compare the logits at *blue* with causal and bidirectional permissions, using the same weights in both stacks. Explain which invariance is guaranteed by the mask and why a permitted path need not produce a large difference for every choice of parameters.

## 11. Inspect a training run

Run the example module with `--train`. For each family, record the number of graded labels and the initial and final loss. Compare `selected_predictions` with `selected_targets` in the printed output. Explain what this demonstrates about the computation. Design a new example that would be needed to begin checking generalization, and explain why the training losses alone cannot rank the three families.

## 12. Check both sides of an encoder--decoder (optional)

With a small random model in evaluation mode, change a later decoder input and compare earlier logits. Next, change a valid source token while keeping the decoder prefix fixed. Finally, append a padding source position and compare the original outputs with those from the unpadded source. State the source validity masks explicitly. Explain the different invariance expectations for these three interventions.

## 13. Add a task head (optional)

Use `encode_hidden` to obtain the small token model's final position vectors before its vocabulary projection. Use a task that fits the teaching vocabulary, such as classifying whether a sentence contains *red* or *blue*. Include exactly one of those colors per sentence, vary the other words independently of the label, and reserve different sentence arrangements for training, validation, and testing. Choose a pooling rule, such as the last nonpadding position in a causal stack, and add a linear head with one output per class. Start the two comparisons from independent copies of the same model weights and the same newly initialized head weights, with a fresh optimizer for each. Train only the head in one copy and fine-tune the whole network in the other, using the same training and validation split. Select settings using validation results, then evaluate on held-out test examples. To study transfer, first pretrain on a separate text set and initialize both copies from that same checkpoint. With random starting weights alone, the experiment compares trainable components rather than pretrained transfer.

## 14. Design a fair comparison (optional)

Choose one downstream task that all three families could solve. Specify the training and held-out data, input formatting, output interpretation, adaptation method, compute budget, and task metric. Explain how you would prevent repeated test-set inspection from influencing model selection. List at least three reasons that raw pretraining losses would be insufficient evidence for declaring a winner.

[Complete worked solutions and reproduction instructions](../../solutions/ch07_pretraining/README.md).
