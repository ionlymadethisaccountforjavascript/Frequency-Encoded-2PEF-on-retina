
"""Small convenience wrapper for NNMF unmixing used in the demo.

Tries to use sklearn.decomposition.NMF if available; otherwise uses a simple multiplicative-update implementation.
"""
import numpy as np
try:
    from sklearn.decomposition import NMF as sklearn_NMF
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

def run_nnmf(V, n_components=2, max_iter=200):
    """V: nonnegative data matrix (n_features x n_samples).
    returns W (n_features x k), H (k x n_samples)
    """
    V = np.asarray(V, dtype=float)
    if HAVE_SKLEARN:
        model = sklearn_NMF(n_components=n_components, init='nndsvda', max_iter=max_iter)
        W = model.fit_transform(V)
        H = model.components_
        return W, H
    # Simple multiplicative update implementation
    eps = 1e-9
    n_features, n_samples = V.shape
    W = np.random.rand(n_features, n_components)
    H = np.random.rand(n_components, n_samples)
    for i in range(max_iter):
        H *= (W.T @ V) / (W.T @ W @ H + eps)
        W *= (V @ H.T) / (W @ (H @ H.T) + eps)
    return W, H
