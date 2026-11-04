def causal_example():
    # Query i can use key j exactly when j <= i.
    allowed = np.tril(np.ones((Q.shape[-2], K.shape[-2]), dtype=bool))
    output, weights = scaled_dot_product_attention(Q, K, V, allowed)
    return output, weights
