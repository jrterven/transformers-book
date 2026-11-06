def residue_composition(sequence):
    """Return residue fractions in AMINO_ACIDS order."""
    ids = _residue_ids(sequence)
    return np.bincount(ids, minlength=len(AMINO_ACIDS)) / len(ids)
