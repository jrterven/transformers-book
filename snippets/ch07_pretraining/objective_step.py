def objective_step(model, optimizer, inputs, labels):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(*inputs)
    loss = selected_loss(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()
