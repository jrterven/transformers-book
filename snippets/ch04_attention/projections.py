H = np.array(
    [[1, 0, 1, 0], [0, 1, 0, 2], [1, 1, 1, 1]], dtype=np.float64
)
W_Q = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.float64)
W_K = W_Q.copy()
W_V = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=np.float64)

Q = H @ W_Q
K = H @ W_K
V = H @ W_V
