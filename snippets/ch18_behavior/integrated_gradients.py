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
