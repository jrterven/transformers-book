"""Small CPU PyTorch examples of adapting a model to a task."""

import math
from operator import index

import torch
from torch import nn
from torch.nn import functional as F

from examples.ch07_pretraining.data import IGNORE, PAD
from examples.ch07_pretraining.model import selected_loss


class TinyClassifier(nn.Module):
    """A deterministic encoder and task head for the supervised comparison."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 3, dtype=torch.float64)
        self.head = nn.Linear(3, 2, dtype=torch.float64)
        with torch.no_grad():
            self.encoder.weight.copy_(self.encoder.weight.new_tensor([
                [1, 0], [0, 1], [0.5, -0.5]
            ]))
            self.encoder.bias.zero_()
            self.head.weight.zero_()
            self.head.bias.zero_()

    def forward(self, features):
        return self.head(torch.tanh(self.encoder(features)))


# snippet:train_classifier
def train_classifier(model, features, labels, *, freeze_encoder, steps=40, lr=0.2):
    """Use the same supervised objective with two trainable parameter sets."""
    steps = index(steps)
    if steps < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("steps and learning rate must be positive")
    model.train()
    model.encoder.requires_grad_(not freeze_encoder)
    model.head.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    if freeze_encoder:
        model.encoder.eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(features), labels)
        loss.backward()
        optimizer.step()
# end-snippet:train_classifier


def _response_inputs(ids, response_mask, pad_id):
    if ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[1] < 2:
        raise ValueError("ids need shape [batch, time] with time at least two")
    if ids.dtype != torch.long or response_mask.dtype != torch.bool:
        raise ValueError("ids must be long and response_mask boolean")
    if ids.shape != response_mask.shape or ids.device != response_mask.device:
        raise ValueError("ids and response_mask must share shape and device")
    valid = ids.ne(pad_id)
    if (ids < 0).any() or not valid[:, 0].all() or (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("each row needs nonnegative IDs in a right-padded prefix")
    if response_mask[:, 0].any():
        raise ValueError("the first token has no earlier logit row to score it")


# snippet:response_labels
def response_labels(ids, response_mask, pad_id=PAD):
    """Use a mask on unshifted tokens to select next-token targets."""
    pad_id = index(pad_id)
    if pad_id < 0:
        raise ValueError("pad_id must be nonnegative")
    _response_inputs(ids, response_mask, pad_id)
    targets = ids[:, 1:]
    selected = response_mask[:, 1:] & targets.ne(pad_id)
    if not selected.any(dim=1).all():
        raise ValueError("each row needs at least one response target")
    return targets.masked_fill(~selected, IGNORE)
# end-snippet:response_labels


def _shifted_logits(logits, labels):
    if logits.ndim != 3 or not logits.is_floating_point():
        raise ValueError("logits need floating shape [batch, time, vocabulary]")
    shifted = logits[:, :-1]
    if (min(shifted.shape) < 1 or shifted.shape[:2] != labels.shape
            or labels.dtype != torch.long or labels.device != logits.device):
        raise ValueError("labels must be long and align with the shifted logits")
    selected = labels.ne(IGNORE)
    if not selected.any(dim=1).all():
        raise ValueError("each row needs at least one response target")
    if ((labels[selected] < 0) | (labels[selected] >= logits.shape[-1])).any():
        raise ValueError("a response label is outside the vocabulary")
    if not torch.isfinite(shifted).all():
        raise ValueError("logits at next-token positions must be finite")
    if shifted.dtype in (torch.float16, torch.bfloat16):
        shifted = shifted.float()
    return shifted


# snippet:completion_logprob
def response_loss(logits, labels):
    """Mean negative log-probability over all scored response tokens."""
    return selected_loss(_shifted_logits(logits, labels), labels)


def completion_logprob(logits, labels):
    """Sum selected token log-probabilities separately for each response."""
    shifted = _shifted_logits(logits, labels)
    log_probs = F.log_softmax(shifted, dim=-1)
    selected = labels.ne(IGNORE)
    safe_labels = labels.masked_fill(~selected, 0)
    token_logp = log_probs.gather(-1, safe_labels[..., None]).squeeze(-1)
    return token_logp.masked_fill(~selected, 0.0).sum(dim=-1)
# end-snippet:completion_logprob


# snippet:lora_linear
class LoRALinear(nn.Module):
    """Keep a Linear base frozen and train a low-rank weight update."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        rank = index(rank)
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be a torch.nn.Linear layer")
        if not 1 <= rank <= min(base.in_features, base.out_features):
            raise ValueError("rank must fit both weight dimensions")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be positive and finite")
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = self.alpha / rank
        self.a = nn.Parameter(base.weight.new_empty((rank, base.in_features)))
        self.b = nn.Parameter(base.weight.new_zeros((base.out_features, rank)))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.a), self.b)

    def merged_weight(self):
        return self.base.weight + self.scale * (self.b @ self.a)
# end-snippet:lora_linear


# snippet:dpo_loss
def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta):
    """Compare policy preference margins relative to a frozen reference."""
    pairs = (policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    if (policy_chosen.numel() == 0
            or any(x.shape != policy_chosen.shape or not x.is_floating_point() for x in pairs)
            or any(not torch.isfinite(x).all() for x in pairs)):
        raise ValueError("log-probabilities must have matching shapes and be finite")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")

    policy_chosen, policy_rejected, reference_chosen, reference_rejected = (
        x.float() if x.dtype in (torch.float16, torch.bfloat16) else x for x in pairs
    )
    policy_gap = policy_chosen - policy_rejected
    reference_gap = reference_chosen.detach() - reference_rejected.detach()
    margin = beta * (policy_gap - reference_gap)
    if not torch.isfinite(margin).all():
        raise FloatingPointError("DPO margin overflowed")
    return F.softplus(-margin).mean()
# end-snippet:dpo_loss
