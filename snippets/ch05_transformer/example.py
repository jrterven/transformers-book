def full_model_example():
    torch.manual_seed(7)
    model = Transformer(12, 12, d_model=4, n_heads=2, d_ff=8,
                        n_encoder_layers=1, n_decoder_layers=1).double()
    source_ids, target_ids, labels = example_batch()
    model.eval()  # Dropout off; autograd remains enabled.
    logits, attention = model(source_ids, target_ids, return_weights=True)
    loss = F.cross_entropy(logits.reshape(-1, 12), labels.reshape(-1),
                           ignore_index=0)
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert model.source_embedding.weight.grad.abs().sum() > 0
    check_prefix_and_padding(model, source_ids, target_ids)
    return logits.detach(), attention, loss.detach()
