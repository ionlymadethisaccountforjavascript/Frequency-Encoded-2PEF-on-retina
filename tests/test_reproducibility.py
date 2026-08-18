"""Tests for deterministic phantom generation under a fixed random seed."""

import numpy as np

from fe2pef_retina.phantoms import rpe_mosaic_phantom


def test_rpe_phantom_repeats_with_same_seed() -> None:
    """Check that identical random seeds reproduce identical retinal maps."""

    first, labels_first = rpe_mosaic_phantom(
        (40, 40), 1.0, 10, 5, 0.3, np.random.default_rng(123)
    )
    second, labels_second = rpe_mosaic_phantom(
        (40, 40), 1.0, 10, 5, 0.3, np.random.default_rng(123)
    )
    assert np.array_equal(labels_first, labels_second)
    assert np.allclose(first, second)
