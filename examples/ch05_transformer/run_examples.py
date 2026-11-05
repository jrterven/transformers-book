"""Run with: python -m examples.ch05_transformer.run_examples."""

import torch
from torch.nn import functional as F

from .model import (
    MultiHeadAttention, PositionalEncoding, PositionwiseFeedForward, Transformer,
)


def toy_multihead():
    """Return the book's deterministic, unmasked two-head calculation."""
    dtype = torch.float64
    x = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 2], [1, 1, 1, 1]], dtype=dtype)[None]
    w_v = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1],
                        [1, 0, 0, 0], [0, 1, 0, 0]], dtype=dtype)
    w_o = torch.tensor([[1, 0, .5, 0], [0, 1, 0, .5],
                        [.5, 0, 1, 0], [0, .5, 0, 1]], dtype=dtype)
    module = MultiHeadAttention(4, 2).to(dtype=dtype)
    with torch.no_grad():
        module.q_proj.weight.copy_(torch.eye(4, dtype=dtype))
        module.k_proj.weight.copy_(torch.eye(4, dtype=dtype))
        # Linear stores W.T for a row-vector mapping x @ W.
        module.v_proj.weight.copy_(w_v.T)
        module.out_proj.weight.copy_(w_o.T)
        output, weights = module(x, x, x)
        heads = weights @ module.split_heads(module.v_proj(x))
        concatenated = module.merge_heads(heads)
    return output, weights, concatenated


def toy_feed_forward():
    """The separate hand-worked FFN: W1=[I,-I], W2=[I;I], zero biases."""
    module = PositionwiseFeedForward(4, 8).double()
    identity = torch.eye(4, dtype=torch.float64)
    x = torch.tensor([1., -2., 0., 3.], dtype=torch.float64)
    with torch.no_grad():
        module.expand.weight.copy_(torch.cat((identity, -identity), dim=0))
        module.expand.bias.zero_()
        module.contract.weight.copy_(torch.cat((identity, identity), dim=1))
        module.contract.bias.zero_()
        hidden = torch.relu(module.expand(x))
        output = module(x)
    return hidden, output


# snippet:vocabulary_readout
def vocabulary_readout_example():
    """A separate three-entry vocabulary: el, coche, EOS.

    The state and readout weights are chosen by hand, not learned.
    """
    state = torch.tensor([[1., -1., 0., 2.]], dtype=torch.float64)
    w_vocab = torch.tensor([
        [1., 0., -1.], [0., -1., 0.], [0., 0., 1.], [.5, 0., .5],
    ], dtype=torch.float64)
    bias = torch.zeros(3, dtype=torch.float64)
    logits = state @ w_vocab + bias
    probabilities = torch.softmax(logits, dim=-1)
    return logits, probabilities
# end-snippet:vocabulary_readout


def example_batch():
    """0=PAD, 1=BOS, 2=EOS; remaining ids are illustrative vocabulary entries.

    Decoder inputs are shifted right relative to labels. The last position
    of the shorter example is padding and is excluded from the objective.
    """
    source_ids = torch.tensor([[3, 4, 5, 2, 0], [6, 5, 2, 0, 0]])
    target_ids = torch.tensor([[1, 8, 9, 10], [1, 11, 9, 0]])
    labels = torch.tensor([[8, 9, 10, 2], [11, 9, 2, 0]])
    return source_ids, target_ids, labels


def check_prefix_and_padding(model, source_ids, target_ids):
    """Check the fixed teaching batch returned by example_batch().

    This demonstration helper assumes pad_id=0, the example's token positions,
    and its 12-entry vocabularies (including replacement IDs 6 and 11). Stack
    depth may vary. For a different batch or vocabulary, adapt the positions,
    replacement IDs, and validity masks rather than reusing this helper as a
    generic validator. It switches the supplied model to evaluation mode.
    """
    model.eval()
    with torch.no_grad():
        baseline = model(source_ids, target_ids)
        prefix = model(source_ids, target_ids[:, :2])
        torch.testing.assert_close(prefix, baseline[:, :2], atol=1e-12, rtol=0)
        changed_target = target_ids.clone()
        changed_target[0, 2] = 6
        changed = model(source_ids, changed_target)
        torch.testing.assert_close(changed[:, :2], baseline[:, :2], atol=1e-12, rtol=0)
        source_valid = source_ids != 0
        changed_source = source_ids.clone()
        changed_source[~source_valid] = 11
        changed = model(changed_source, target_ids, source_valid=source_valid)
        torch.testing.assert_close(changed, baseline, atol=1e-12, rtol=0)


# snippet:example
def full_model_example():
    torch.manual_seed(7)
    model = Transformer(12, 12, d_model=4, n_heads=2, d_ff=8,
                        n_encoder_layers=1, n_decoder_layers=1).double()
    source_ids, target_ids, labels = example_batch()
    model.eval()  # Dropout off; autograd remains enabled.
    logits, attention = model(source_ids, target_ids, return_weights=True)
    loss = F.cross_entropy(logits.reshape(-1, 12), labels.reshape(-1),
                           ignore_index=0)
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert model.source_embedding.weight.grad.abs().sum() > 0
    check_prefix_and_padding(model, source_ids, target_ids)
    return logits.detach(), attention, loss.detach()
# end-snippet:example


def main():
    torch.set_num_threads(1)
    torch.set_printoptions(precision=6, sci_mode=False)
    output, weights, concatenated = toy_multihead()
    print("Two-head weights [head, query, key]:\n", weights[0])
    print("Concatenated heads:\n", concatenated[0])
    print("After W_O:\n", output[0])
    pe = PositionalEncoding(4).double().pe[0, :3]
    print("Sinusoidal positions 0, 1, 2:\n", pe)
    hidden, ffn_output = toy_feed_forward()
    print("Toy FFN ReLU hidden:\n", hidden)
    print("Toy FFN output:\n", ffn_output)
    readout_logits, readout_probabilities = vocabulary_readout_example()
    print("Separate three-word readout [el, coche, EOS]:")
    print("  Logits:", readout_logits[0])
    print("  Vocabulary probabilities:", readout_probabilities[0])
    torch.testing.assert_close(
        readout_logits, torch.tensor([[2., 1., 0.]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        readout_probabilities.sum(-1), torch.ones(1, dtype=torch.float64)
    )
    logits, attention, loss = full_model_example()
    print("Logits shape:", tuple(logits.shape))
    for kind, values in attention.items():
        print(f"{kind} attention shape:", tuple(values[0].shape))
    print(f"Illustrative valid-token cross-entropy: {loss.item():.6f}")
    print("Passed: finite gradients, prefix/future invariance, source-padding isolation.")
    print("Randomly initialized model: these values do not measure translation quality.")


if __name__ == "__main__":
    main()
