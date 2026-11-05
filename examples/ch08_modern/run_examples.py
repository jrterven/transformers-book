"""Work through three components, then check a complete decoder on a CPU."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from examples.ch07_pretraining.data import IGNORE, PAD, VOCAB, causal_pairs, encode
from examples.ch07_pretraining.model import selected_loss

from .model import ModernDecoder, SwiGLU, rms_norm


def normalization_example():
    x = torch.tensor([3.0, 4.0], dtype=torch.float64)
    return {
        "input": x.tolist(),
        "layer_norm": F.layer_norm(x, (2,), eps=0.0).tolist(),
        "rms_norm": rms_norm(x, torch.ones_like(x), eps=0.0).tolist(),
    }


@torch.no_grad()
def swiglu_example():
    layer = SwiGLU(4, 2).double()
    x = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    layer.gate_proj.weight.zero_()
    layer.gate_proj.weight[1, 0] = 2.0
    layer.up_proj.weight.zero_()
    layer.up_proj.weight[:, 0] = x.new_tensor([3.0, -1.0])
    layer.down_proj.weight.copy_(x.new_tensor([
        [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]
    ]))
    g, u = layer.gate_proj(x), layer.up_proj(x)
    return {
        "gate_projection": g.tolist(),
        "up_projection": u.tolist(),
        "product": (F.silu(g) * u).tolist(),
        "output": layer(x).tolist(),
    }


def rotation_example():
    # Both unrotated vectors are (1, 0). Their rotations are (cos a, sin a).
    positions = torch.tensor([[3, 1], [5, 3]], dtype=torch.float64)
    angles = positions * (math.pi / 6)
    vectors = torch.stack((angles.cos(), angles.sin()), dim=-1)
    scores = (vectors[:, 0] * vectors[:, 1]).sum(dim=-1)
    return {
        "positions": positions.int().tolist(),
        "angle_step_degrees": 30,
        "rotated_query_key_pairs": vectors.tolist(),
        "pairwise_dot_products": scores.tolist(),
    }


def prepared_batch():
    rows = [causal_pairs(encode(text))
            for text in ("the red car stopped", "the blue moved")]
    inputs, labels = zip(*rows)
    return (pad_sequence(inputs, batch_first=True, padding_value=PAD),
            pad_sequence(labels, batch_first=True, padding_value=IGNORE))


# snippet:example_checks
def one_update(model, ids, labels):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    before = selected_loss(model(ids), labels)
    before.backward()
    finite_gradients = all(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in model.parameters()
    )
    if not finite_gradients:
        raise FloatingPointError("each parameter must have a finite gradient")
    optimizer.step()
    with torch.no_grad():
        after = selected_loss(model(ids), labels)
    return {"loss_before": before.item(), "loss_after": after.item(),
            "finite_gradients": finite_gradients}
# end-snippet:example_checks


def decoder_example():
    torch.manual_seed(81)
    model = ModernDecoder(vocab_size=len(VOCAB), pad_id=PAD).double()
    ids, labels = prepared_batch()
    changed = ids.clone()
    changed[0, -1] = encode("moved")[0]
    extended = F.pad(ids, (0, 1), value=PAD)
    with torch.no_grad():
        logits = model(ids)
        prefix_change = (model(changed)[0, :-1] - logits[0, :-1]).abs().max()
        real = ids != PAD
        pad_change = (model(extended)[:, :ids.shape[1]][real] -
                      logits[real]).abs().max()
        shift_change = (model(ids, start_pos=7) - logits).abs().max()
    return {
        "seed": 81,
        "inputs": ids.tolist(),
        "labels": labels.tolist(),
        "scored_labels": int((labels != IGNORE).sum()),
        "logits_shape": list(logits.shape),
        "parameters_per_block": sum(p.numel() for p in model.blocks[0].parameters()),
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "earlier_logits_max_change": prefix_change.item(),
        "appended_padding_max_change": pad_change.item(),
        "shared_position_shift_max_change": shift_change.item(),
        "update": one_update(model, ids, labels),
    }


def run():
    with torch.random.fork_rng(devices=[]):
        return {
            "normalization": normalization_example(),
            "swiglu": swiglu_example(),
            "rotation": rotation_example(),
            "decoder": decoder_example(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    args = parser.parse_args()
    torch.set_num_threads(1)
    text = json.dumps(run(), indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
