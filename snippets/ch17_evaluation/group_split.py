def split_by_group(group_ids, group_counts, *, seed=17):
    """Return train, validation, and test row indices with disjoint groups."""
    group_ids = _group_ids(group_ids)
    names = np.unique(group_ids)
    train_count, validation_count, test_count = map(index, group_counts)
    counts = (train_count, validation_count, test_count)
    if min(counts) < 1 or sum(counts) != len(names):
        raise ValueError("three positive group counts must sum to the number of groups")

    shuffled = np.random.default_rng(seed).permutation(names)
    parts = np.split(shuffled, [train_count, train_count + validation_count])
    return tuple(np.flatnonzero(np.isin(group_ids, part)) for part in parts)
