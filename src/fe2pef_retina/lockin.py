"""Reference generation and phase-aware digital lock-in demodulation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .config import LaserConfig


def reference_waveform(
    channel: str,
    time_s: np.ndarray,
    laser1: LaserConfig,
    laser2: LaserConfig,
) -> np.ndarray:
    """Generate the phase-aware reference for one lock-in channel.

    What happens in this function:
    1. Laser modulation phases are reconstructed from frequency and phase settings.
    2. Fundamental channels use sine references matching the intensity tags.
    3. Mixed and harmonic channels use cosine references from product identities.
    4. Signs are selected so ideal pathway amplitudes are positive.
    """

    time_s = np.asarray(time_s, dtype=float)
    phase1 = 2.0 * np.pi * laser1.modulation_frequency_hz * time_s + laser1.phase_rad
    phase2 = 2.0 * np.pi * laser2.modulation_frequency_hz * time_s + laser2.phase_rad
    if channel == "dc":
        return np.ones_like(time_s)
    if channel == "f1":
        return np.sin(phase1)
    if channel == "f2":
        return np.sin(phase2)
    if channel == "difference":
        return np.cos(phase1 - phase2)
    if channel == "sum":
        return -np.cos(phase1 + phase2)
    if channel == "2f1":
        return -np.cos(2.0 * phase1)
    if channel == "2f2":
        return -np.cos(2.0 * phase2)
    raise ValueError(f"unknown lock-in channel: {channel}")


def build_reference_matrix(
    channels: Iterable[str],
    time_s: np.ndarray,
    laser1: LaserConfig,
    laser2: LaserConfig,
) -> np.ndarray:
    """Build normalized linear weights for simultaneous lock-in demodulation.

    What happens in this function:
    1. One reference waveform is generated for every requested channel.
    2. DC weights perform a temporal mean.
    3. AC weights use twice the reference divided by sample count.
    4. Matrix multiplication then demodulates many pixels simultaneously.
    """

    time_s = np.asarray(time_s, dtype=float)
    if time_s.ndim != 1 or time_s.size < 2:
        raise ValueError("time_s must be a one-dimensional array with at least two samples")
    weights = []
    for channel in channels:
        reference = reference_waveform(channel, time_s, laser1, laser2)
        if channel == "dc":
            weights.append(reference / time_s.size)
        else:
            weights.append(2.0 * reference / time_s.size)
    return np.vstack(weights)


def demodulate_time_series(
    signal: np.ndarray,
    time_s: np.ndarray,
    channels: Iterable[str],
    laser1: LaserConfig,
    laser2: LaserConfig,
) -> np.ndarray:
    """Demodulate one or many temporal fluorescence signals.

    What happens in this function:
    1. The final signal axis is interpreted as time.
    2. Normalized lock-in weights are generated for all channels.
    3. A tensor contraction projects each signal onto every reference.
    4. The first output axis indexes channels.
    """

    signal = np.asarray(signal, dtype=float)
    if signal.shape[-1] != len(time_s):
        raise ValueError("the final signal dimension must match time_s")
    weights = build_reference_matrix(channels, time_s, laser1, laser2)
    return np.tensordot(weights, signal, axes=(1, signal.ndim - 1))


def orthogonality_matrix(
    channels: Iterable[str],
    time_s: np.ndarray,
    laser1: LaserConfig,
    laser2: LaserConfig,
) -> np.ndarray:
    """Measure finite-window orthogonality of the requested references.

    What happens in this function:
    1. Raw reference waveforms are assembled as matrix rows.
    2. Every row is normalized to unit norm.
    3. The Gram matrix reveals leakage caused by insufficient or noninteger cycles.
    4. An ideal acquisition has an approximately identity Gram matrix.
    """

    references = np.vstack(
        [reference_waveform(channel, time_s, laser1, laser2) for channel in channels]
    )
    norms = np.linalg.norm(references, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normalized = references / norms
    return normalized @ normalized.T
