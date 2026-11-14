"""Offline NumPy examples for multimodal representations."""

import math
from operator import index

import numpy as np

from examples.ch01_language.model import token_nll
from examples.ch04_attention.attention import scaled_dot_product_attention


def _real_matrix(values, name, *, allow_empty=False):
    values = np.asarray(values)
    if values.ndim != 2 or (not allow_empty and values.size == 0):
        raise ValueError(f"{name} needs a matrix of feature or score rows")
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return values


def _unit_rows(features):
    features = features.astype(np.float64, copy=False)
    largest = np.max(np.abs(features), axis=1, keepdims=True)
    if not np.isfinite(largest).all() or np.any(largest == 0):
        raise ValueError("each feature row needs a finite nonzero norm")
    # Rescale before taking the norm to avoid squaring very large values.
    with np.errstate(under="ignore"):
        rescaled = features / largest
        return rescaled / np.linalg.norm(rescaled, axis=1, keepdims=True)


# snippet:contrastive_logits
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
# end-snippet:contrastive_logits


# snippet:contrastive_loss
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
# end-snippet:contrastive_loss


# snippet:bidirectional_recall
def _recall_at_k(scores, relevance, k):
    per_query = []
    for row, relevant in zip(scores, relevance):
        total = int(relevant.sum())
        if total == 0:
            per_query.append(None)
            continue
        # Stable sorting keeps candidate order for ties.
        best = sorted(range(len(row)), key=row.__getitem__, reverse=True)[:k]
        per_query.append(int(relevant[best].sum()) / total)
    defined = [value for value in per_query if value is not None]
    return {
        "per_query": per_query,
        "defined_queries": len(defined),
        "mean": float(np.mean(defined)) if defined else None,
    }


def bidirectional_recall_at_k(scores, relevance, k):
    """Average the fraction of each query's relevant items in its top k."""
    scores = _real_matrix(scores, "scores", allow_empty=True)
    relevance = np.asarray(relevance)
    if (relevance.shape != scores.shape or relevance.dtype.kind not in "biu"
            or np.any((relevance != 0) & (relevance != 1))):
        raise ValueError("relevance must be a binary matrix matching the scores")
    k = index(k)
    if k < 1:
        raise ValueError("k must be positive")
    return {
        "image_to_text": _recall_at_k(scores, relevance, k),
        "text_to_image": _recall_at_k(scores.T, relevance.T, k),
    }
# end-snippet:bidirectional_recall


# snippet:image_feature_bridge
def image_feature_bridge(queries, keys, values, projection):
    """Read image features with queries and project the result to LM width."""
    queries = _real_matrix(queries, "queries")
    keys = _real_matrix(keys, "keys")
    values = _real_matrix(values, "values")
    projection = _real_matrix(projection, "projection")
    if projection.shape[0] != values.shape[1]:
        raise ValueError("projection rows must match the value width")
    context, weights = scaled_dot_product_attention(queries, keys, values)
    with np.errstate(over="raise", invalid="raise"):
        language_tokens = context @ projection.astype(np.float64, copy=False)
    return context, weights, language_tokens
# end-snippet:image_feature_bridge
