"""Noise and detector-response helpers for simulated fluorescence data."""

from __future__ import annotations

import numpy as np


def first_order_detector_gain(frequency_hz: float, bandwidth_hz: float) -> float:
    """Return amplitude attenuation of a first-order low-pass detector.

    What happens in this function:
    1. DC is passed without attenuation.
    2. The frequency-to-bandwidth ratio is formed.
    3. The standard first-order magnitude response is returned.
    """

    if frequency_hz < 0 or bandwidth_hz <= 0:
        raise ValueError("frequency must be nonnegative and bandwidth positive")
    return float(1.0 / np.sqrt(1.0 + (frequency_hz / bandwidth_hz) ** 2))


def poisson_gaussian_samples(
    expectation: np.ndarray,
    rng: np.random.Generator,
    read_noise_std: float,
    saturation_counts: float | None = None,
) -> np.ndarray:
    """Draw detector samples with shot noise, read noise, and optional clipping.

    What happens in this function:
    1. Negative expectations are clipped because Poisson rates cannot be negative.
    2. Poisson samples model photon-counting shot noise.
    3. Gaussian samples model additive read/electronic noise.
    4. Optional saturation clips the final detector output.
    """

    expectation = np.clip(np.asarray(expectation, dtype=float), 0.0, None)
    measured = rng.poisson(expectation).astype(float)
    if read_noise_std > 0:
        measured += rng.normal(0.0, read_noise_std, size=measured.shape)
    if saturation_counts is not None:
        measured = np.clip(measured, 0.0, saturation_counts)
    return measured


def lockin_noise_std(
    mean_counts_per_sample: np.ndarray,
    samples_per_dwell: int,
    read_noise_std: float,
    is_dc: bool,
) -> np.ndarray:
    """Approximate noise of an orthogonal lock-in estimator.

    What happens in this function:
    1. Poisson variance is approximated by the mean count per temporal sample.
    2. Read-noise variance is added independently.
    3. Averaging over the dwell reduces variance by the sample count.
    4. AC quadrature projection introduces a factor of two relative to DC averaging.
    """

    if samples_per_dwell <= 0:
        raise ValueError("samples_per_dwell must be positive")
    variance_per_sample = np.clip(mean_counts_per_sample, 0.0, None) + read_noise_std**2
    factor = 1.0 if is_dc else 2.0
    return np.sqrt(factor * variance_per_sample / samples_per_dwell)
