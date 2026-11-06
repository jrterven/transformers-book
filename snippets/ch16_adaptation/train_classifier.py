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
