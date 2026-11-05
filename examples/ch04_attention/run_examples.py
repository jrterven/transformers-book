"""Run from the repository root: python -m examples.ch04_attention.run_examples."""

import numpy as np

from .attention import attention_backward, scaled_dot_product_attention, stable_softmax


# snippet:projections
H = np.array(
    [[1, 0, 1, 0], [0, 1, 0, 2], [1, 1, 1, 1]], dtype=np.float64
)
W_Q = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.float64)
W_K = W_Q.copy()
W_V = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=np.float64)

Q = H @ W_Q
K = H @ W_K
V = H @ W_V
# end-snippet:projections


# snippet:one_query_example
def one_query_example(query_index=0):
    """Follow one token's query before doing every query together."""
    query = Q[query_index]
    scores = np.empty(len(K), dtype=np.float64)
    for j in range(len(K)):
        scores[j] = np.sum(query * K[j]) / np.sqrt(len(query))

    weights = stable_softmax(scores)
    contributions = np.empty_like(V)
    output = np.zeros(V.shape[1], dtype=np.float64)
    for j in range(len(V)):
        contributions[j] = weights[j] * V[j]
        output += contributions[j]
    return scores, weights, contributions, output
# end-snippet:one_query_example


# snippet:batch_heads
def batch_heads_example():
    # Axes: (batch, head, token, feature). Each head attends independently.
    q = np.broadcast_to(Q, (2, 2, 3, 2)).copy()
    k = np.broadcast_to(K, (2, 2, 3, 2)).copy()
    v = np.broadcast_to(V, (2, 2, 3, 2)).copy()
    v[1, 1] *= 2.0
    output, weights = scaled_dot_product_attention(q, k, v)
    return output, weights
# end-snippet:batch_heads


# snippet:causal_example
def causal_example():
    # Query i can use key j exactly when j <= i.
    allowed = np.tril(np.ones((Q.shape[-2], K.shape[-2]), dtype=bool))
    output, weights = scaled_dot_product_attention(Q, K, V, allowed)
    return output, weights
# end-snippet:causal_example


def gradient_max_error(q, k, v, upstream, allowed_mask):
    """Compare the manual backward pass with central finite differences."""
    analytic = attention_backward(q, k, v, upstream, allowed_mask)
    inputs = [q.copy(), k.copy(), v.copy()]
    step = 1e-6
    max_error = 0.0
    for argument, array in enumerate(inputs):
        for index in np.ndindex(array.shape):
            original = array[index]
            array[index] = original + step
            plus, _ = scaled_dot_product_attention(*inputs, allowed_mask)
            array[index] = original - step
            minus, _ = scaled_dot_product_attention(*inputs, allowed_mask)
            array[index] = original
            numerical = np.sum((plus - minus) * upstream) / (2.0 * step)
            error = abs(numerical - analytic[argument][index])
            max_error = max(max_error, error)
    return max_error


def main():
    np.set_printoptions(precision=4, suppress=True)
    output, weights = scaled_dot_product_attention(Q, K, V)
    scores = (Q @ K.T) / np.sqrt(Q.shape[-1])
    for label, value in (("H", H), ("Q", Q), ("K", K), ("V", V)):
        print(f"{label}:\n{value}\n")

    row_scores, row_weights, contributions, row_output = one_query_example()
    print("One query: the first token position")
    for label, value in (("Scaled scores", row_scores),
                         ("Weights over keys", row_weights),
                         ("Weighted value contributions", contributions),
                         ("Their sum: one output vector", row_output)):
        print(f"{label}:\n{value}\n")
    np.testing.assert_allclose(row_scores, scores[0])
    np.testing.assert_allclose(row_weights, weights[0])
    np.testing.assert_allclose(row_output, output[0])

    print("All queries: the same calculation in matrices")
    for label, value in (("Scaled scores", scores), ("Weights", weights),
                         ("Output", output)):
        print(f"{label}:\n{value}\n")

    causal_output, causal_weights = causal_example()
    allowed = np.tril(np.ones((3, 3), dtype=bool))
    print(f"Causal allowed mask (True = allowed):\n{allowed}\n")
    print(f"Causal weights:\n{causal_weights}\n")
    print(f"Causal output:\n{causal_output}\n")

    batched_output, batched_weights = batch_heads_example()
    print(f"Batch/head output shape: {batched_output.shape}")
    print(f"Batch/head weights shape: {batched_weights.shape}")
    np.testing.assert_allclose(batched_output[0, 0], output)
    np.testing.assert_allclose(batched_output[0, 1], output)
    np.testing.assert_allclose(batched_output[1, 0], output)
    np.testing.assert_allclose(batched_output[1, 1], 2.0 * output)

    prefix, _ = scaled_dot_product_attention(Q[:2], K[:2], V[:2], allowed[:2, :2])
    np.testing.assert_allclose(prefix, causal_output[:2])
    future_k, future_v = K.copy(), V.copy()
    future_k[2] += 100.0
    future_v[2] -= 100.0
    changed, _ = scaled_dot_product_attention(Q, future_k, future_v, allowed)
    np.testing.assert_allclose(changed[:2], causal_output[:2])

    permutation = np.array([2, 0, 1])
    permuted, _ = scaled_dot_product_attention(
        Q[permutation], K[permutation], V[permutation]
    )
    np.testing.assert_allclose(permuted, output[permutation])

    upstream = np.array([[0.2, -0.4], [0.5, 0.3], [-0.1, 0.6]])
    error = gradient_max_error(Q, K, V, upstream, allowed)
    if error > 1e-8:
        raise AssertionError(f"gradient check failed: maximum error {error}")
    print(f"Masked backward maximum absolute error: {error:.3e}")
    print("Checks passed: one-query/matrix agreement, head isolation, "
          "causal prefix/future invariance, permutation equivariance, masked backward.")


if __name__ == "__main__":
    main()
