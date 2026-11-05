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
