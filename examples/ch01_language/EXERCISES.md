# Chapter 1 exercises

Tokens, Embeddings, and Neural Language Models

These statements match the revised Chapter 1. Exercises 1-7 build foundational
understanding and can be worked by hand; 8-10 combine reasoning and Python;
11-12 are challenges. Use the chapter's 16-entry word vocabulary and fixed
teaching weights unless an exercise explicitly supplies another tokenizer or
probabilities.

1. **Separate the three representations.** For *the red car*, write the tokens, their IDs, and their four-coordinate embeddings. Explain what information the ID carries and why it is not the same object as the embedding. Give the shape of $E$ and the shape of $X$.

2. **Repeat a token.** Look up the IDs for *red car red*. Write the resulting $3\times4$ matrix. Which rows are equal? Explain why the lookup alone has not made the two occurrences of *red* depend on their different surroundings.

3. **Count tokens, not words.** In a teaching tokenizer, let *unhelpful* split into *un*, *help*, and *ful*, while *helpful* splits into *help* and *ful*. Keep *the* and *car* as whole tokens. Count the tokens in *the unhelpful car* and compare with a word-level tokenizer. Explain why a real BPE tokenizer need not produce these particular splits. Do not try to encode these extra pieces with our 16-entry word vocabulary.

4. **Read the axes.** A batch contains five sentences, each with three tokens, and the embedding width is four. State the shapes of the ID array and its retrieved embeddings. Then take one two-token context from each sentence. For hidden width four and vocabulary size 16, give the batch shapes of the context embeddings, joined vectors, hidden vectors, and logits. Which axis must softmax normalize?

5. **Normalize a full vocabulary.** Using logits 2 for *car*, 1 for *red*, and 0 for each other entry, recalculate $D$ and all 16 probabilities. Verify that they sum to one before rounding. Why does a zero logit receive a positive probability? Is *car* more probable than all the other 15 entries combined?

6. **Grade the observed target.** With *car* as the target, calculate the loss $-\ln p_{\mathrm{car}}$ for the distribution in Exercise 5 and for a uniform distribution over 16 entries. Which has the smaller loss? If the observed target were *red*, which probability would you grade? Explain why a plausible continuation need not be the observed target.

7. **Follow the context calculation.** Calculate each coordinate before and after ReLU in the worked context equation (first embedding + twice the second + bias). Then reverse the two input tokens and repeat the calculation. Use the output weights described in the text to explain the two different probability distributions.

8. **Build the training pairs.** Write the two-token context windows and labels for *the red car*, including the boundary tokens. Identify which context predicts *car* and which predicts `EOS`. Check with `next_token_pairs`. Explain why the target position must not be included in its own context, even though the same word may appear earlier. If generation chooses *bike* after *the red*, what two-token window is used next? Contrast this with the final training window from the reference sentence.

9. **Combine different prediction steps.** Suppose the observed-token probabilities for *the*, *red*, *car*, and `EOS` are $0.5$, $0.25$, $0.8$, and $0.5$. Calculate the probability of that whole sentence and ending, its total negative log-likelihood, and the mean loss over its four targets. Explain why the four probabilities need not sum to one and why multiplying them does not assume independent tokens. Check the arithmetic with NumPy.

10. **Find the window's limitation.** Compare predictions after *the red* and *I saw the red*. Convert each prefix to IDs and pass only its final two IDs, `ids[-2:]`, to `context_logits`. Which IDs reach the network? Compare the complete output distributions. Could training different weights make this deterministic two-token model distinguish these prefixes? Give an example where words farther back would matter.

11. **Challenge: change the numbering.** Exchange the IDs of *the* and *red*. Describe how to rearrange the embedding rows, output-weight columns, output biases, and training targets so that probabilities by token name stay unchanged. Why does changing only the input IDs fail? Verify the full distribution with Python, comparing matching token names rather than unchanged array positions.

12. **Challenge: keep softmax stable.** Add 1000 to every logit from Exercise 5. Explain why the probabilities should not change and check this using `stable_softmax`. Stack the original and shifted rows into a batch and verify that each row sums to one. For logits $(1000,-1000)$ and target ID 1, compare `token_nll` with taking the negative logarithm of the softmax probability. Explain why the direct loss remains finite even if the tiny probability rounds to zero.

[Written solutions (PDF)](../../solutions/ch01_language/written-solutions.pdf) · [Executed notebook](../../solutions/ch01_language/python-solutions.ipynb) · [Notebook PDF](../../solutions/ch01_language/python-solutions.pdf)
