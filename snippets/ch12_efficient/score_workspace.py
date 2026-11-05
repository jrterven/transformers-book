def score_workspace(query_length, key_length, query_tile, key_tile):
    """Return element counts for one dense score array and one largest tile."""
    query_length, key_length, query_tile, key_tile = map(
        index, (query_length, key_length, query_tile, key_tile)
    )
    if min(query_length, key_length, query_tile, key_tile) < 1:
        raise ValueError("sequence lengths and tile sizes must be positive")
    full = query_length * key_length
    largest_tile = min(query_length, query_tile) * min(key_length, key_tile)
    return full, largest_tile
