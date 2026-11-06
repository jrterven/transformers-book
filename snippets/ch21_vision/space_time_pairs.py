def space_time_pairs(frames, patches_per_frame):
    """Count ordered attention pairs, including self-pairs, with no CLS row."""
    frames, patches_per_frame = map(index, (frames, patches_per_frame))
    if min(frames, patches_per_frame) < 1:
        raise ValueError("frame and patch counts must be positive")
    spatial = frames * patches_per_frame**2
    temporal = patches_per_frame * frames**2
    return {
        "joint": (frames * patches_per_frame) ** 2,
        "spatial": spatial,
        "temporal": temporal,
        "divided": spatial + temporal,
    }
