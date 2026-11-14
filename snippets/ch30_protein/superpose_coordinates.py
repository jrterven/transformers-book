def superpose_coordinates(mobile, reference):
    """Rotate and translate matched row vectors with the Kabsch method."""
    mobile, reference = _matched_coordinates(mobile, reference)
    moving = mobile - mobile.mean(axis=0)
    center = reference.mean(axis=0)
    target = reference - center
    left, _, right_t = np.linalg.svd(moving.T @ target)
    # The last axis absorbs a possible reflection from the SVD.
    handedness = np.linalg.det(left @ right_t)
    rotation = left @ np.diag([1, 1, np.sign(handedness)]) @ right_t
    return moving @ rotation + center
