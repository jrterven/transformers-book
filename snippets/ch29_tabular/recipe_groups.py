def recipe_groups(features):
    """Group exact ingredient vectors and keep curing age as a feature."""
    _, group = np.unique(features[:, :7], axis=0, return_inverse=True)
    return group
