def train_linear_probe(features, labels, *, steps=120, lr=0.5):
    """Fit only a two-class affine head on fixed feature rows."""
    features, labels = _probe_inputs(features, labels)
    steps = index(steps)
    if steps < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("steps and learning rate must be positive")

    weight = np.zeros((features.shape[1], 2))
    bias = np.zeros(2)
    target = np.eye(2)[labels]
    with np.errstate(over="raise", invalid="raise"):
        for _ in range(steps):
            probabilities = stable_softmax(features @ weight + bias)
            difference = (probabilities - target) / len(features)
            weight -= lr * (features.T @ difference)
            bias -= lr * difference.sum(axis=0)
    return weight, bias
