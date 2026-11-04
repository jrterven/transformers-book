"""Token-weighted training and evaluation for the Chapter 6 experiment."""

import math
from pathlib import Path

import torch
from torch.nn import functional as F

from examples.ch05_transformer.model import Transformer
from .data import EOS, PAD, VOCAB, collate
from .decoding import greedy_decode


# snippet:token_loss
def token_loss(logits, targets):
    count = (targets != PAD).sum()
    if count.item() == 0:
        raise ValueError("a loss needs at least one non-padding target")
    total = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
        ignore_index=PAD, reduction="sum",
    )
    return total, count
# end-snippet:token_loss


# snippet:train_step
def train_step(model, optimizer, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch.source, batch.decoder_input)
    total, count = token_loss(logits, batch.target)
    loss = total / count
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(gradient_norm):
        raise ValueError("non-finite gradient norm")
    optimizer.step()
    return loss.item()
# end-snippet:train_step


def exact_matches(generated, targets):
    """Count full generated targets, including EOS; ignore trailing PAD."""
    def trim(row):
        ids = row.tolist()
        while ids and ids[-1] == PAD:
            ids.pop()
        return ids
    count = 0
    for prediction, reference in zip(generated, targets):
        prediction, reference = trim(prediction), trim(reference)
        count += bool(prediction and prediction[-1] == EOS and prediction == reference)
    return count


@torch.no_grad()
def evaluate(model, sequences, batch_size=64):
    if not sequences or batch_size < 1:
        raise ValueError("evaluation needs data and a positive batch_size")
    # snippet:evaluate
    model.eval()
    loss_sum = token_count = correct_tokens = correct_sequences = terminated = 0
    for start in range(0, len(sequences), batch_size):
        batch = collate(sequences[start:start + batch_size])
        logits = model(batch.source, batch.decoder_input)
        total, count = token_loss(logits, batch.target)
        valid = batch.target != PAD
        loss_sum += total.item()
        token_count += count.item()
        correct_tokens += ((logits.argmax(-1) == batch.target) & valid).sum().item()
        generated = greedy_decode(model, batch.source,
                                  max_new_tokens=batch.target.shape[1] + 1)
        correct_sequences += exact_matches(generated, batch.target)
        terminated += generated.eq(EOS).any(dim=1).sum().item()
    nll = loss_sum / token_count
    return {"nll": nll, "perplexity": math.exp(nll),
            "token_accuracy": correct_tokens / token_count,
            "exact_match": correct_sequences / len(sequences),
            "termination_rate": terminated / len(sequences),
            "tokens": token_count, "sequences": len(sequences)}
    # end-snippet:evaluate


# snippet:checkpoint
def save_checkpoint(path, model, model_config, experiment, step, validation):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "model_config": model_config, "vocabulary": list(VOCAB),
                "experiment": experiment, "step": step,
                "validation": validation}, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["vocabulary"] != list(VOCAB):
        raise ValueError("checkpoint vocabulary does not match this example")
    model = Transformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.eval(), checkpoint
# end-snippet:checkpoint
