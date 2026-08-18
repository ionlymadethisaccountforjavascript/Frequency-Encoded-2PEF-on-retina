"""Tests for calibrated and blind nonnegative unmixing."""

import numpy as np

from fe2pef_retina.unmixing import calibrated_nnls_unmix, regularized_nmf


def test_calibrated_nnls_recovers_exact_mixture() -> None:
    """Check exact recovery for a noiseless full-rank nonnegative mixture."""

    signature = np.array([[1.0, 0.2], [0.3, 1.0], [0.6, 0.4]])
    maps = np.array([[[1.0, 0.2], [0.0, 0.5]], [[0.1, 0.8], [1.0, 0.2]]])
    signal = (signature @ maps.reshape(2, -1)).reshape(3, 2, 2)
    recovered = calibrated_nnls_unmix(signal, signature)
    assert np.allclose(recovered, maps, atol=1e-10)


def test_regularized_nmf_reconstructs_low_rank_signal() -> None:
    """Check that regularized NMF obtains a low residual on positive synthetic data."""

    rng = np.random.default_rng(5)
    signature = rng.random((4, 2)) + 0.2
    concentrations = rng.random((2, 30))
    signal = signature @ concentrations
    result = regularized_nmf(
        signal,
        components=2,
        alpha_l1=0.0,
        alpha_l2=0.0,
        iterations=2000,
        restarts=4,
        tolerance=1e-9,
        rng=np.random.default_rng(6),
    )
    relative_residual = np.linalg.norm(signal - result.signature @ result.concentrations) / np.linalg.norm(signal)
    assert relative_residual < 0.03
