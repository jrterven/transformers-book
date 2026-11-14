def mean_pool_residues(embeddings, residue_mask):
    """Average only the embedding rows marked as residues."""
    embeddings = np.asarray(embeddings)
    residue_mask = np.asarray(residue_mask)
    if (embeddings.ndim != 2 or min(embeddings.shape) == 0
            or embeddings.dtype.kind not in "iuf"):
        raise ValueError("embeddings need a nonempty real [positions, width] matrix")
    if residue_mask.shape != embeddings.shape[:1] or residue_mask.dtype != np.bool_:
        raise ValueError("provide one Boolean mask value per embedding row")
    if not residue_mask.any():
        raise ValueError("select at least one residue row")
    with np.errstate(over="raise", invalid="raise"):
        residues = embeddings[residue_mask].astype(np.float64, copy=False)
        if not np.isfinite(residues).all():
            raise ValueError("selected residue embeddings must be finite")
        return residues.mean(axis=0)
