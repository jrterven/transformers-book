def history_counts(stores):
    """Count the persistent state or cache scalars for each layer."""
    return tuple(
        store.size if isinstance(store, np.ndarray)
        else sum(array.size for array in store)
        for store in stores
    )
