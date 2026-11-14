"""Small, reusable calculations for the interpretability experiments."""

import torch
from torch import nn


def ridge_probe(train, target, test, regularization=1.0):
    """Fit a centered linear readout using only the supplied training rows."""
    mean = train.mean(dim=0)
    inputs = train - mean
    offset = target.mean(dim=0)
    kernel = inputs @ inputs.T
    system = kernel + regularization * torch.eye(len(train), dtype=train.dtype, device=train.device)
    coefficients = torch.linalg.solve(system, target - offset)
    return (test - mean) @ (inputs.T @ coefficients) + offset


# snippet:integrated_gradients
def integrated_gradients(score, baseline, target, steps):
    """Integrate gradients along a straight embedding path by midpoint rule."""
    if steps < 1 or baseline.shape != target.shape:
        raise ValueError("positive steps and paired embeddings are required")
    delta = target.detach() - baseline.detach()
    accumulated = torch.zeros_like(delta)
    with torch.enable_grad():
        for index in range(steps):
            fraction = (index + 0.5) / steps
            sample = (baseline.detach() + fraction * delta).requires_grad_(True)
            gradient, = torch.autograd.grad(score(sample), sample)
            accumulated += gradient
    return delta * (accumulated / steps)
# end-snippet:integrated_gradients


class SparseDictionary(nn.Module):
    """Learn a sparse code for centered residual vectors with MSE reconstruction."""

    def __init__(self, width, features):
        super().__init__()
        self.encoder = nn.Linear(width, features)
        self.decoder = nn.Parameter(torch.randn(features, width))
        self.b_dec = nn.Parameter(torch.zeros(width))

    def forward(self, inputs):
        code = self.encoder(inputs - self.b_dec).relu()
        directions = torch.nn.functional.normalize(self.decoder, dim=-1)
        return code, code @ directions + self.b_dec


def fit_dictionary(train, test, features=1536, sparsity=0.05, updates=160):
    """Fit on normalized training activations and report held-out diagnostics."""
    center = train.mean(dim=0)
    rms = (train - center).square().mean().sqrt()
    train, test = (train - center) / rms, (test - center) / rms
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(18)
        model = SparseDictionary(train.shape[1], features)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        for _ in range(updates):
            batch = train[torch.randint(len(train), (32,))]
            code, recovered = model(batch)
            loss = (recovered - batch).square().mean() + sparsity * code.sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        code, reconstructed = model(test)
        # Scales are relative to the same zero (training-mean) prediction.
        return {
            "features": features, "l1_weight": sparsity, "updates": updates,
            "test_rows": len(test),
            "mean_active": float((code > 0).sum(-1).float().mean()),
            "relative_squared_error": float((test - reconstructed).square().sum() / test.square().sum()),
            "test_features_ever_active": int((code > 0).any(dim=0).sum()),
        }
