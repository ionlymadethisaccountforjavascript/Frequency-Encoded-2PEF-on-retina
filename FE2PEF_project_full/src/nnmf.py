import numpy as np

def nnmf(V, n_components=2, n_iter=200, eps=1e-9):
    rng = np.random.RandomState(0)
    n_features, n_samples = V.shape
    W = rng.rand(n_features, n_components) + 1e-3
    H = rng.rand(n_components, n_samples) + 1e-3
    for i in range(n_iter):
        H *= (W.T @ V) / (W.T @ W @ H + eps)
        W *= (V @ H.T) / (W @ H @ H.T + eps)
        W = W / (np.maximum(W.sum(axis=0), eps))
    return W, H
