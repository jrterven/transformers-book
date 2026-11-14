def patch_embeddings(patches, weight, *, cls=None, positions=None):
    """Project every patch, then optionally add a CLS row and positions."""
    patches = _real_array(patches, "patches")
    weight = _real_array(weight, "weight")
    if (patches.ndim != 2 or weight.ndim != 2 or min(patches.shape) < 1
            or min(weight.shape) < 1 or patches.shape[1] != weight.shape[0]):
        raise ValueError("patch and projection widths must match")

    with np.errstate(over="raise", invalid="raise"):
        tokens = patches @ weight
        if cls is not None:
            cls = _real_array(cls, "CLS")
            if cls.shape != (tokens.shape[1],):
                raise ValueError("CLS needs one vector at the model width")
            tokens = np.concatenate((cls[None, :], tokens), axis=0)
        if positions is not None:
            positions = _real_array(positions, "positions")
            if positions.shape != tokens.shape:
                raise ValueError("provide one positional vector per token")
            tokens = tokens + positions
    return tokens
