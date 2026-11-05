"""Teaching calculations for dense budgets and a top-1 expert bank."""

import math

import numpy as np

from examples.ch04_attention.attention import stable_softmax


BILLION = 10**9


# snippet:compute_budget
def training_flops(parameters, tokens):
    """Estimate dense training FLOPs from parameter and token counts."""
    if not (math.isfinite(parameters) and parameters > 0
            and math.isfinite(tokens) and tokens > 0):
        raise ValueError("parameters and tokens must be finite and positive")
    return 6 * parameters * tokens


def tokens_at_budget(flops, parameters):
    """Solve C = 6ND for D, before rounding to whole training tokens."""
    if not (math.isfinite(flops) and flops > 0):
        raise ValueError("flops must be finite and positive")
    return flops / training_flops(parameters, 1)
# end-snippet:compute_budget


# snippet:allocation_sweep
def allocation_sweep(flops, model_sizes_b):
    """Return (parameters in billions, tokens in billions, toy loss)."""
    rows = []
    for n_b in model_sizes_b:
        d_b = tokens_at_budget(flops, n_b * BILLION) / BILLION
        loss = 1 + 0.4 / math.sqrt(n_b) + 2 / math.sqrt(d_b)
        rows.append((n_b, d_b, loss))
    return rows
# end-snippet:allocation_sweep


# snippet:top1_routing
def route_top1(logits, capacity):
    """Return probabilities, zero-based expert IDs, gates, and acceptance."""
    logits = np.asarray(logits)
    if logits.ndim != 2 or min(logits.shape) == 0:
        raise ValueError("logits must have shape [tokens, experts]")
    if (isinstance(capacity, (bool, np.bool_))
            or not isinstance(capacity, (int, np.integer)) or capacity < 0):
        raise ValueError("capacity must be a nonnegative integer")

    probabilities = stable_softmax(logits)
    selected = probabilities.argmax(axis=1)
    gates = probabilities[np.arange(len(selected)), selected]
    accepted = np.zeros(len(selected), dtype=bool)
    for expert in range(probabilities.shape[1]):
        requests = np.flatnonzero(selected == expert)
        accepted[requests[:capacity]] = True
    return probabilities, selected, gates, accepted
# end-snippet:top1_routing


# snippet:expert_branch
def expert_branch(x, selected, gates, accepted, w_up, w_down):
    """Compute accepted updates and place them back in token order."""
    x = np.asarray(x, dtype=np.float64)
    w_up = np.asarray(w_up, dtype=np.float64)
    w_down = np.asarray(w_down, dtype=np.float64)
    if x.ndim != 2 or w_up.ndim != 3:
        raise ValueError("x needs [tokens, width] and w_up [experts, width, hidden]")
    experts, width, hidden = w_up.shape
    if x.shape[1] != width or w_down.shape != (experts, hidden, width):
        raise ValueError("input and expert widths must agree")
    if any(array.shape != (len(x),) for array in (selected, gates, accepted)):
        raise ValueError("routing arrays need one entry per token")
    if np.any((selected < 0) | (selected >= experts)):
        raise ValueError("each selected ID must index an expert")

    branch = np.zeros_like(x)
    for expert in range(experts):
        rows = np.flatnonzero((selected == expert) & accepted)
        if rows.size == 0:
            continue
        hidden_rows = np.maximum(x[rows] @ w_up[expert], 0)
        branch[rows] = gates[rows, None] * (hidden_rows @ w_down[expert])
    return branch
# end-snippet:expert_branch
