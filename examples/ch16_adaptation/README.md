# Chapter 16. From Pretrained Weights to Task Behavior

Run the CPU examples from the repository root with PyTorch.

```sh
python -m pip install -r requirements-ch05.txt
python -m examples.ch16_adaptation.run_examples --output output/ch16.json
python -m unittest discover -s tests -p test_adaptation.py -v
```

## Choose which parameters learn

The classifier starts from one fixed encoder and one zero-initialized
head. Both runs use the same eight training points, cross-entropy
objective, and 40 full-batch SGD steps. One trains only the head, and
the other trains the encoder and head together. The report counts
trainable parameters, checks the encoder's weight change, and evaluates
both runs on the same four held-out points.

## Choose which tokens receive a loss

Pass token IDs `[B,T]` and a Boolean mask in the same unshifted layout to
`response_labels`. Mark the response content and its EOS token. The
function shifts the IDs and mask together and ignores padding.

Pass the original logits `[B,T,V]` and those shifted labels `[B,T-1]` to
both scoring functions. `response_loss` averages negative log-probability
over scored tokens. `completion_logprob` returns a separate sum of log
probabilities for each response, including its EOS when selected by the
mask. Prompt tokens still provide causal context to the model.

## Update a weight with LoRA

`LoRALinear` freezes the supplied base layer and adds a trainable branch
with matrices `a[r,d_in]` and `b[d_out,r]` and scale `alpha/r`. The
initial `b` is zero, so the starting output equals the base output. The
merged weight has the same shape as the base. Keep the base bias when
applying it with a single linear operation.

The example saves the two adapter tensors, rank, and scale parameter to
an in-memory checkpoint. It reconstructs the same base layer before
restoring the adapter. For an adapter trained on a pretrained model,
record the base model revision and the names of the adapted layers so
the factors can be attached to the intended weights.

## Compare a policy with a fixed reference

`dpo_loss` takes four sets of completion log-probabilities. It compares
the policy's chosen-versus-rejected gap with the corresponding gap from
a frozen reference. A positive `beta` scales that difference before
`softplus` computes the loss. The reference scores are detached.


## Exercises and complete solutions

The [twelve exercises](EXERCISES.md) progress from hand calculations to
tensor checks, checkpoint reconstruction and an evaluation protocol.
The [solutions guide](../../solutions/ch16_adaptation/README.md) links the
executed notebook and both PDF editions; each covers every exercise.

Keep three distinctions visible when running the examples. Freezing a
weight leaves its operation available to backpropagation; `no_grad` can
break that path. Ignoring a prompt target removes its direct loss while
the prompt still supplies context. DPO compares response probability
ratios relative to a reference, so a common shift of both policy scores
leaves a pair's loss unchanged.

`response_loss` uses a mean over all selected tokens in the batch. Longer
responses therefore supply more token contributions. `completion_logprob`
returns one sum per response, from which a different, equal-response
reduction can be built explicitly. The helpers require a right-padded
prefix, a distinct padding ID and at least one selected target per row;
they do not create the model's causal attention mask.

LoRA's rank constrains the update, not the full merged matrix. Random `a`
and zero `b` initially preserve the base output while letting `b` receive
a gradient. Both factors initialized to zero would stall ordinary gradient
descent. The wrapper has no adapter dropout or nonlinear bottleneck.
