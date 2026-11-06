# Chapter 2 | Exercises

Recurrence and Encoder-Decoder Models

Exercises 1-10 build understanding with small calculations; 11-13 use the companion code; 14 is a reasoning challenge.

## 1. Separate the state from the rule

Compare the prefixes *red green blue* and *yellow green blue* when the task is to report the first color. Explain why a two-token window cannot distinguish them. What could make their recurrent states different? Which quantities change while reading one sequence, which change in a training update, and which encoder quantities are reset before an unrelated source sequence?

## 2. Follow a state by hand

Starting at $h_0=0$, calculate the weighted sum and updated state at each position for inputs $(0.2,0.4,-0.1)$ using $h_i=\tanh(0.5h_{i-1}+x_i)$. Keep full precision between updates and display six decimals. Explain why the final state is positive despite the negative last input. Compare with `rnn_trace`.

## 3. Reverse the order

Repeat Exercise 2 for $(-0.1,0.4,0.2)$. Explain why the middle input gives a different state even though its value is unchanged. Then set the recurrent weight to zero: which inputs can affect the final state now?

## 4. Keep, write, and expose

Begin with $c_{i-1}=0.5$, $f_i=0.8$, $i_i=0.5$, $g_i=-0.4$, and $o_i=0.6$. Calculate $c_i$ and $h_i$. Change only $f_i$ to 0, then to 1. Restore the original values and close only the input gate, then restore them and close only the output gate. Calculate both states each time. Does a zero hidden value imply an empty cell? Treat gate values 0 and 1 as idealized limits.

## 5. Compare two gradient paths

Calculate $0.5^{20}$ and $0.99^{20}$. Explain which path each describes, and why the first is an upper bound for our scalar RNN rather than its exact derivative. Why does the second not guarantee that every gradient in an LSTM remains large? Distinguish a small state value from a small derivative.

## 6. Pass the right state

For a one-layer, forward-only encoder with batch size five and hidden width 64, give the shapes of its final hidden and cell states. Identify where each goes in the decoder and whether this operation also copies the encoder's weights. Explain why passing only the last source token's embedding is different, even when two sources end with the same word.

## 7. Shift before grading

Write decoder inputs and labels for source *red blue green yellow* in the reversal task. Include `BOS` and `EOS`. At the step graded against *blue*, which token is the current input? If the preceding generated token were *white* instead of the reference *green*, what would the next input be during teacher forcing and during generation?

## 8. Count prediction positions

Batch sources of lengths three and five. Give the shapes of source IDs, source embeddings, decoder inputs, labels, logits, and each final encoder state for this chapter's model. Count real target labels, including endings. Why is the loss denominator 10 rather than 12? If the two sentence-mean losses were 0.2 and 0.8, calculate their token-weighted batch mean and their unweighted sentence mean.

## 9. Grade the ending

Recalculate the sequence probability and mean loss from correct-token probabilities $(0.8,0.6,0.75,0.9)$. Change only the last probability, assigned to `EOS`, to 0.1 and repeat. How much does the mean loss increase? Explain why these four probabilities need not add to one, and why the ending contributes to the loss.

## 10. Locate the information path

While the decoder predicts *coche*, how can it use the source word *car*? Draw the path through the encoder's state pair, the decoder's first two updates, and the output layer. Does the decoder directly reread the source embedding at this point? Describe what additional source access could make this task easier.

## 11. Test padding

Append three right-padding positions to one source while keeping its true length unchanged. Compare its encoded hidden and cell states and its evaluation-mode decoder logits. Then bypass packing and run the encoder directly on the two dense embedding arrays. Explain why zero padding embeddings can change recurrent states even though correct packing leaves the original computation unchanged.

## 12. Separate scoring from generation

On a held-out color sequence, print the reference, the highest-scoring tokens under teacher forcing, and the freely generated sequence. Align their decoder inputs step by step and locate the first divergence, if one occurs. If every inspected generation is correct, use an untrained model or a shorter training run in a separate output directory. Retain any generated `PAD` or `BOS` before the first `EOS`. The program saves only its selected `best.pt`, not every intermediate checkpoint.

## 13. Read a length experiment carefully

Run the training example and report held-out exact-match counts and rates for each source length, along with each group's size. Which data selected the checkpoint? Explain why the default split does not test unseen lengths. Design a separate experiment with training, validation, and test lengths specified in advance. If you implement it, use a common generation cap large enough for the longest evaluation source plus `EOS`, rather than the default cap of seven.

## 14. Challenge: greedy or most probable?

Fix the source and consider a tiny decoder. Its first token is $A$ with probability 0.6 or $B$ with probability 0.4. After a first $A$, it assigns 0.51 to `EOS` and 0.49 to $B$. After a first $B$, it assigns 0.99 to `EOS` and 0.01 to $A$. If it emits a second ordinary token, it emits `EOS` next with probability 1. All other choices have probability zero. List the four complete sequences and their probabilities. Which does greedy decoding produce, and which has the largest complete-sequence probability? Explain the difference.

[Worked solutions, notebook, and PDFs](../../solutions/ch02_recurrence/README.md).
