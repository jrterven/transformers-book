# Chapter 16 exercises

From Pretrained Weights to Task Behavior

Exercises 1-4 use paper calculations, 5-10 use PyTorch, 11 reconstructs a checkpoint, and 12 designs an adaptation comparison. All twelve [worked solutions](../../solutions/ch16_adaptation/README.md) are provided in every edition.

Conventions: column-vector LoRA is $y=W_0x+(\alpha/r)BAx$, with $A$ shaped `[r,d_in]` and $B$ shaped `[d_out,r]`. Response masks mark unshifted target tokens; the helpers shift targets and mask together. DPO uses $m=\beta[(\log p_+-\log p_-)-(\log p_{\mathrm{ref},+}-\log p_{\mathrm{ref},-})]$ and loss $-\log\sigma(m)$.

The frozen-operation example in Exercise 9 uses a trainable scalar `u` followed by a frozen multiplier 3: `h = u*x`, `y = 3*h`, `loss = (y-target)**2/2`. At `u=1`, `x=2` and `target=0`, the gradient with respect to `u` is 36.

1. **Count a task head.** A frozen encoder returns six features. You add a linear head for three classes, with one bias per class. Count the trainable parameters. If the encoder has 120 parameters, give the trainable count for head-only training and for full fine-tuning.

2. **Mark the response targets.** Use the packed sequence $(`BOS`,`Question`,`Assistant`,`New`,`Address`,`EOS`,`PAD`)$. Treat each item as one token and the first three as the prompt. Write the shifted input and target rows. Mark the targets with weight one and identify the input position whose logits predict `New`.

3. **Follow a rank-one update.** Let $W_0=I_2$, $A=(2,1)$, $B=(1,-1)^\top$, $r=1$, $\alpha=0.5$, and $\mathbf{x}=(1,2)^\top$. Compute the base output, the scaled adapter output, and their sum. Then form the merged matrix and use it to check the output.

4. **Compute a relative preference.** Use policy log-probabilities $(-2,-3.5)$ and reference log-probabilities $(-3,-3.25)$, with the chosen response first in each pair. Compute the two gaps and the DPO margin at $\beta=1$. Write the loss in terms of the sigmoid. Subtract 0.5 from both policy scores and recompute the margin. Explain how this changes the loss.

5. **Check target alignment in code.** Use the seven tokens from Exercise 2 with IDs $(1,3,4,5,6,2,0)$. Put them in a long tensor of shape $[1,7]$. Build a Boolean response mask of that shape, including `EOS` in the response. Call `response_labels`. Identify each surviving label and the logit row with which it aligns.

6. **Perturb a logit row.** Use the IDs and labels from Exercise 5 and a float tensor of zeros of shape $[1,7,7]$ for the logits. Call `response_loss` and `completion_logprob`. Increase the `Question` logit at `BOS` by 2 and repeat. Restart from zero logits and increase the `New` logit at `Assistant` by 2. Explain which perturbation changes the scored outputs.

7. **Compare two batch averages.** One response has two selected targets, each with loss 1. Another has six selected targets, each with loss 3. Compute the mean over all selected tokens and the mean of the two response means. Verify both using `response_loss` and `completion_logprob`: construct a batch with vocabulary size two and target token ID 1. At a selected row, logits $(\ln(\exp(\ell)-1),0)$ give target loss $\ell$. Use padding ID 0 and right padding. Explain which response receives more weight under the helper's reduction, and why adding extra padding changes neither result.

8. **Inspect a fresh adapter.** Construct a PyTorch linear layer with four input features, three outputs, and a bias. Wrap it in `LoRALinear` with rank two and $\alpha=4$. Use `base.weight.new_tensor([1, 0, 0, 1])` as the input. Compare the base and wrapped outputs before training. List the shapes and gradient flags of the base weight, bias, and both factors. Verify the trainable count by summing the sizes of tensors with `requires_grad`. Then inspect the scalar branch $y=w_0x+sba x$ with $w_0=1$, $x=2$, $s=2$, $a=1$, $b=0$, target zero, and $L=y^2/2$. Calculate both factor gradients and take one SGD step with learning rate $0.1$. Recompute the gradients. What happens if both factors start at zero?

9. **Compare trainable weights.** From `run_examples.py`, import `classification_data`. Take its first two outputs as the training features and labels. Call `train_classifier` on two fresh `TinyClassifier` instances with `steps=1`, freezing the encoder in one run. Compare encoder and head parameters to their initial values. Repeat from fresh instances with `steps=2`. How does the zero initial head affect the comparison? Explain why the comparison uses the same initial weights and data. Separately implement the scalar example $y=3ux$ from the text. Freeze the multiplier and verify that $u$ still receives gradient 36 at $u=1$, $x=2$, and target zero. Explain why wrapping that multiplication in `no_grad` would give a different result.

10. **Keep the reference fixed (optional).** Build four scalar PyTorch tensors with `requires_grad=True`, using the log-probabilities from Exercise 4. Pass them to `dpo_loss` with `beta=1` and call `backward` on the loss. Inspect each tensor's gradient. Explain which part of the loss produces policy gradients and how the code treats the reference tensors.

11. **Rebuild an adapted model (optional).** Imagine an adapter was trained on the attention query and value projections, plus a new classifier. List the arrays and settings you would save for a colleague who already has access to the base. Design a small output comparison that could catch an incorrect base revision or an adapter loaded into the wrong projection. For a two-input affine layer, implement probes at zero and at both coordinate unit vectors. Show why one probe alone can miss an incorrect weight matrix. For the complete model, distinguish this numerical check from verification of the base revision and projection names.

12. **Plan an adaptation comparison (optional).** Compare training a head on a frozen encoder, full fine-tuning, and training a low-rank adapter plus a head on labeled address requests. Describe the output interface you would give all three systems. Specify what you would hold fixed, how you would split repeated templates, and which accuracy, parameter, and storage measurements you would report. Distinguish trainable scalars, checkpoint bytes, and peak training memory. Include an unchanged pretrained baseline and describe how you would repeat and select runs without using test results.
