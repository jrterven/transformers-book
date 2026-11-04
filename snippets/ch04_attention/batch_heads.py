def batch_heads_example():
    # Axes: (batch, head, token, feature). Each head attends independently.
    q = np.broadcast_to(Q, (2, 2, 3, 2)).copy()
    k = np.broadcast_to(K, (2, 2, 3, 2)).copy()
    v = np.broadcast_to(V, (2, 2, 3, 2)).copy()
    v[1, 1] *= 2.0
    output, weights = scaled_dot_product_attention(q, k, v)
    return output, weights
