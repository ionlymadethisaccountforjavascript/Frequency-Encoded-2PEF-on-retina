"""
Phase 3: Signal Unmixing (The Matrix Factorization Engine)
==============================================================
Step 5: Non-Negative Matrix Factorization (NNMF)
Step 6: Final Image Reconstruction

Takes the three component images (I1, I2, I3 -- one per demodulated
frequency, each shaped (ny, nx)) and factorizes them into two
crosstalk-free fluorophore abundance images.
"""

import numpy as np
from sklearn.decomposition import NMF


def unmix_components(I1, I2, I3, n_components=2, random_state=0, max_iter=2000,
                      ref_coeffs_A=None, ref_coeffs_B=None):
    """
    Step 5: Reshape (ny, nx) component images into a (n_pixels, 3) data
    matrix where each row is a pixel's frequency profile [I1, I2, I3], then
    factorize as V ~= W @ H, with:
      W: (n_pixels, 2) -- crosstalk-free abundance of fluorophore A and B per pixel
      H: (2, 3)        -- the recovered mixing matrix (each row = a fluorophore's
                           response coefficients across the 3 pathways)

    Returns (abundance_A_image, abundance_B_image, H, model).
    """
    ny, nx = I1.shape
    V = np.stack([I1.ravel(), I2.ravel(), I3.ravel()], axis=1)  # (n_pixels, 3)
    V = np.clip(V, 0, None)  # enforce non-negativity going in, as required by NNMF

    model = NMF(n_components=n_components, init="nndsvda", random_state=random_state,
                max_iter=max_iter, solver="mu")
    W = model.fit_transform(V)   # (n_pixels, 2)
    H = model.components_        # (2, 3)

    # NNMF component ordering/scale is arbitrary; identify which output
    # column best corresponds to "fluorophore A-like" vs "B-like" by matching
    # each recovered H-row against the KNOWN reference coefficient vectors
    # using full-vector cosine similarity (not just a single pathway value --
    # comparing only H[:,0] mis-assigns components whenever a fluorophore's
    # peak pathway isn't pathway 1, which happens for realistic, more
    # spectrally-distinct coefficient choices). We try both possible
    # assignments and keep whichever maximizes total similarity.
    if ref_coeffs_A is None or ref_coeffs_B is None:
        from forward_model import FocalVolumeMixer
        ref_coeffs_A = FocalVolumeMixer.COEFFS_A2E if ref_coeffs_A is None else ref_coeffs_A
        ref_coeffs_B = FocalVolumeMixer.COEFFS_LIPOFUSCIN if ref_coeffs_B is None else ref_coeffs_B

    def cos_sim(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)

    score_AB = cos_sim(H[0], ref_coeffs_A) + cos_sim(H[1], ref_coeffs_B)  # H0->A, H1->B
    score_BA = cos_sim(H[1], ref_coeffs_A) + cos_sim(H[0], ref_coeffs_B)  # H1->A, H0->B

    if score_AB >= score_BA:
        idx_A, idx_B = 0, 1
    else:
        idx_A, idx_B = 1, 0

    abundance_A = W[:, idx_A].reshape(ny, nx)
    abundance_B = W[:, idx_B].reshape(ny, nx)

    H_ordered = H[[idx_A, idx_B], :]
    return abundance_A, abundance_B, H_ordered, model


def reconstruct_images(abundance_A, abundance_B, normalize=True):
    """Step 6: final reshaping/normalization for display or saving."""
    A = abundance_A.copy()
    B = abundance_B.copy()
    if normalize:
        A = A / (A.max() + 1e-9)
        B = B / (B.max() + 1e-9)
    return A, B
