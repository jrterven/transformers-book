def decode_with_context(model, previous, state, context):
    embedding = model.target_embedding(previous)
    hidden, _ = model.decoder(torch.cat((embedding, context), dim=-1), state)
    return model.output(hidden)
