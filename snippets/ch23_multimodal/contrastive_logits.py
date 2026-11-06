def contrastive_logits(image_features, text_features, scale):
    """Return scaled similarities with image rows and text columns."""
    images = _real_matrix(image_features, "image features")
    texts = _real_matrix(text_features, "text features")
    if images.shape[1] != texts.shape[1]:
        raise ValueError("image and text features must share a width")
    scale = np.asarray(scale)
    if scale.ndim != 0 or scale.dtype.kind not in "iuf":
        raise ValueError("scale must be a positive finite scalar")
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite scalar")
    images, texts = _unit_rows(images), _unit_rows(texts)
    with np.errstate(over="raise", invalid="raise"):
        return scale * (images @ texts.T)
