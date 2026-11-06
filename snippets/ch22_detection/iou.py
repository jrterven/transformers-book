def box_iou(a, b):
    """Compare positive-area boxes with continuous corner coordinates."""
    a, b = _boxes([a, b])
    with np.errstate(over="raise", under="raise", invalid="raise"):
        lo = np.maximum(a[:2], b[:2])
        hi = np.minimum(a[2:], b[2:])
        intersection = np.maximum(0, hi - lo).prod()
        area_a = (a[2:] - a[:2]).prod()
        area_b = (b[2:] - b[:2]).prod()
        union = area_a + (area_b - intersection)
    return float(intersection / union)


def mask_iou(a, b):
    """Compare Boolean masks on the same pixel grid."""
    a, b = map(np.asarray, (a, b))
    if a.ndim != 2 or a.shape != b.shape or a.dtype != np.bool_ or b.dtype != np.bool_:
        raise ValueError("masks must be Boolean arrays on the same 2D grid")
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return intersection / union if union else None
