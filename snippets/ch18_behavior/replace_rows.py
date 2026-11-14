def replace_rows(output, positions, donor_vectors):
    """In each batch row replace one token's whole block-output vector."""
    patched = output.clone()
    batch = torch.arange(output.shape[0], device=output.device)
    patched[batch, positions, :] = donor_vectors
    return patched
