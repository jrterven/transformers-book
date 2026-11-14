def symmetric_contrastive_loss(logits):
    """Return image-to-text, text-to-image, and mean diagonal cross-entropies."""
    logits = _real_matrix(logits, "logits")
    if logits.shape[0] != logits.shape[1]:
        raise ValueError("diagonal training targets need a square batch")
    targets = np.arange(len(logits))
    # token_nll subtracts the row maximum before calculating log softmax.
    with np.errstate(over="ignore", under="ignore"):
        image_to_text = float(np.sum(token_nll(logits, targets) / len(logits)))
        text_to_image = float(np.sum(token_nll(logits.T, targets) / len(logits)))
    if not (math.isfinite(image_to_text) and math.isfinite(text_to_image)):
        raise FloatingPointError("diagonal cross-entropy exceeds float64 range")
    return image_to_text, text_to_image, 0.5 * image_to_text + 0.5 * text_to_image
