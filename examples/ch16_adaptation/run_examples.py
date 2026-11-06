"""Run the Chapter 16 task adaptation examples without network access."""

import argparse
from copy import deepcopy
import io
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .model import (
    LoRALinear,
    TinyClassifier,
    completion_logprob,
    dpo_loss,
    response_labels,
    response_loss,
    train_classifier,
)


def classification_data():
    train = torch.tensor([
        [-2, -1], [-1.5, 0], [-1, 1], [-0.8, -0.5],
        [0.8, 0.5], [1, -1], [1.5, 0], [2, 1],
    ], dtype=torch.float64)
    held_out = torch.tensor([
        [-1.25, 0.5], [-0.6, -1], [0.6, 1], [1.25, -0.5],
    ], dtype=torch.float64)
    return train, (train[:, 0] > 0).long(), held_out, (held_out[:, 0] > 0).long()


def parameter_count(model, *, trainable_only=False):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)


@torch.no_grad()
def classification_metrics(model, features, labels):
    model.eval()
    logits = model(features)
    return {
        "loss": float(F.cross_entropy(logits, labels)),
        "accuracy": float((logits.argmax(dim=-1) == labels).double().mean()),
    }


def classifier_example():
    train, labels, held_out, held_out_labels = classification_data()
    initial = TinyClassifier()
    reports = {}
    for name, freeze in (("frozen_encoder", True), ("full_fine_tuning", False)):
        model = deepcopy(initial)
        before = classification_metrics(model, held_out, held_out_labels)
        train_classifier(model, train, labels, freeze_encoder=freeze)
        encoder_change = max(
            float((p.detach() - old.detach()).abs().max())
            for p, old in zip(model.encoder.parameters(), initial.encoder.parameters())
        )
        reports[name] = {
            "trainable_parameters": parameter_count(model, trainable_only=True),
            "total_parameters": parameter_count(model),
            "max_encoder_change": encoder_change,
            "held_out_before": before,
            "held_out_after": classification_metrics(model, held_out, held_out_labels),
        }
    return reports


def response_example_inputs():
    # BOS, User, Red?, Assistant, Yes, EOS, PAD
    ids = torch.tensor([[1, 3, 4, 5, 6, 2, 0]])
    mask = torch.tensor([[False, False, False, False, True, True, False]])
    logits = torch.zeros((1, ids.shape[1], 7), dtype=torch.float64)
    logits[0, 3, 6] = math.log(6)
    logits[0, 4, 2] = math.log(3)
    return ids, mask, logits


def response_example():
    ids, mask, logits = response_example_inputs()
    labels = response_labels(ids, mask)
    return {
        "ids": ids[0].tolist(),
        "unshifted_response_mask": mask[0].tolist(),
        "shifted_labels": labels[0].tolist(),
        "completion_log_probability": float(completion_logprob(logits, labels)[0]),
        "mean_response_loss": float(response_loss(logits, labels)),
    }


@torch.no_grad()
def worked_base():
    base = nn.Linear(2, 2, bias=False, dtype=torch.float64)
    base.weight.copy_(base.weight.new_tensor([[1, 2], [0, -1]]))
    return base


@torch.no_grad()
def checkpoint_roundtrip(layer):
    # Recreate the same frozen base before loading the adapter factors.
    stream = io.BytesIO()
    torch.save({
        "rank": layer.rank, "alpha": layer.alpha,
        "a": layer.a.detach(), "b": layer.b.detach(),
    }, stream)
    stream.seek(0)
    saved = torch.load(stream, map_location="cpu", weights_only=True)
    restored = LoRALinear(worked_base(), saved["rank"], saved["alpha"])
    restored.a.copy_(saved["a"])
    restored.b.copy_(saved["b"])
    return restored


@torch.no_grad()
def lora_example():
    x = torch.tensor([3.0, 1.0], dtype=torch.float64)
    layer = LoRALinear(worked_base(), rank=1, alpha=2)
    base_output = layer.base(x)
    initial_output = layer(x)
    initial_b_is_zero = bool(torch.count_nonzero(layer.b) == 0)
    layer.a.copy_(layer.a.new_tensor([[1, -1]]))
    layer.b.copy_(layer.b.new_tensor([[0.5], [1]]))
    output = layer(x)
    merged = F.linear(x, layer.merged_weight(), layer.base.bias)
    restored = checkpoint_roundtrip(layer)
    larger = LoRALinear(nn.Linear(8, 6), rank=2, alpha=4)
    return {
        "scale": layer.scale,
        "base_output": base_output.tolist(),
        "initial_b_is_zero": initial_b_is_zero,
        "initial_output": initial_output.tolist(),
        "scaled_update": (output - base_output).tolist(),
        "adapted_output": output.tolist(),
        "merged_output": merged.tolist(),
        "restored_output": restored(x).tolist(),
        "rank_of_update": int(torch.linalg.matrix_rank(layer.b @ layer.a)),
        "larger_base_parameters": parameter_count(larger.base),
        "larger_adapter_parameters": parameter_count(larger, trainable_only=True),
    }


def preference_example():
    policy = torch.tensor([-3.0, -4.0], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([-4.0, -4.5], dtype=torch.float64)
    beta = 2
    loss = dpo_loss(policy[0], policy[1], reference[0], reference[1], beta)
    loss.backward()
    return {
        "policy": policy.detach().tolist(),
        "frozen_reference": reference.tolist(),
        "beta": beta,
        "margin": float(beta * ((policy[0] - policy[1]) - (reference[0] - reference[1])).detach()),
        "loss": float(loss.detach()),
        "policy_gradients": policy.grad.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {
        "classification": classifier_example(),
        "response_targets": response_example(),
        "lora": lora_example(),
        "preference": preference_example(),
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
