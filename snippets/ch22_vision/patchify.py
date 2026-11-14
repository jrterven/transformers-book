def patchify(image, patch_size):
    """Flatten an HWC image into nonoverlapping patches in row-major order."""
    image = np.asarray(image)
    patch_size = index(patch_size)
    if image.ndim != 3 or min(image.shape) < 1:
        raise ValueError("image needs nonempty shape [height, width, channels]")
    if image.dtype.kind not in "iuf" or not np.isfinite(image).all():
        raise ValueError("image pixels must be finite real numbers")
    height, width, channels = image.shape
    if patch_size < 1 or height % patch_size or width % patch_size:
        raise ValueError("a positive patch size must divide both image dimensions")

    rows, cols = height // patch_size, width // patch_size
    grid = image.reshape(rows, patch_size, cols, patch_size, channels)
    grid = grid.transpose(0, 2, 1, 3, 4)
    return grid.reshape(rows * cols, patch_size**2 * channels)
