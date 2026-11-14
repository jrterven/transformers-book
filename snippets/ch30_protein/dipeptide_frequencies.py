def dipeptide_frequencies(sequence):
    """Return adjacent ordered-pair frequencies, or zeros for one residue."""
    ids = _residue_ids(sequence)
    width = len(AMINO_ACIDS)
    if len(ids) == 1:
        return np.zeros((width, width))
    pair_ids = width * ids[:-1] + ids[1:]
    counts = np.bincount(pair_ids, minlength=width**2).reshape(width, width)
    return counts / len(pair_ids)
