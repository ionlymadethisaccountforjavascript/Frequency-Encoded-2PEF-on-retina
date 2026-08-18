"""Conventional single-detector and emission-filtered TPEF baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DetectorConfig
from .noise import lockin_noise_std


def load_emission_matrix(
    path: str | Path,
    species: Iterable[str],
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Load a detector-by-species emission mixing matrix from CSV.

    What happens in this function:
    1. Long-format rows are read with detector, species, response, and provenance.
    2. Requested species are kept in the exact project order.
    3. A pivot table forms detector rows and species columns.
    4. Nonnegative finite responses are validated before return.
    """

    frame = pd.read_csv(path)
    required = {"detector", "species", "response", "provenance"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"emission table is missing columns: {sorted(missing)}")
    requested = list(species)
    selected = frame[frame["species"].isin(requested)].copy()
    pivot = selected.pivot(index="detector", columns="species", values="response")
    missing_species = [name for name in requested if name not in pivot.columns]
    if missing_species:
        raise ValueError(f"emission table is missing species: {missing_species}")
    pivot = pivot[requested]
    matrix = pivot.to_numpy(dtype=float)
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("emission responses must be finite and nonnegative")
    return matrix, list(pivot.index), selected


def conventional_single_detector_image(
    concentration_maps: np.ndarray,
    species_brightness: np.ndarray,
) -> np.ndarray:
    """Create an unfiltered conventional TPEF intensity image.

    What happens in this function:
    1. Each concentration map is multiplied by its relative brightness.
    2. Species contributions are summed because one detector cannot label their origin.
    3. The result is useful for visualization but cannot uniquely recover species maps.
    """

    maps = np.asarray(concentration_maps, dtype=float)
    brightness = np.asarray(species_brightness, dtype=float)
    if maps.shape[0] != brightness.size:
        raise ValueError("species_brightness must contain one value per species")
    return np.tensordot(brightness, maps, axes=(0, 0))


def simulate_emission_filtered_channels(
    concentration_maps: np.ndarray,
    emission_matrix: np.ndarray,
    optical_throughput: float,
    detector: DetectorConfig,
    rng: np.random.Generator,
    subtract_known_background: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Simulate conventional multi-detector emission-filtered TPEF.

    What happens in this function:
    1. Emission filters mix species into detector channels through a response matrix.
    2. Optical throughput reduces signal to represent filter and beam-splitter losses.
    3. Temporal averaging produces one mean count estimate per detector and pixel.
    4. Shot noise, background shot noise, and read noise are approximated analytically.
    5. Known mean background may be subtracted while stochastic noise remains.
    """

    maps = np.asarray(concentration_maps, dtype=float)
    mixing = np.asarray(emission_matrix, dtype=float)
    if mixing.shape[1] != maps.shape[0]:
        raise ValueError("emission matrix columns must equal species count")
    if not 0 < optical_throughput <= 1:
        raise ValueError("optical_throughput must lie in (0, 1]")

    spatial_shape = maps.shape[1:]
    flat_maps = maps.reshape(maps.shape[0], -1)
    scale = detector.photons_per_relative_unit_per_sample
    expected_signal = scale * optical_throughput * (mixing @ flat_maps)
    background = detector.background_counts_per_sample + detector.dark_counts_per_sample
    expected_total = np.clip(expected_signal + background, 0.0, None)
    n_samples = max(1, int(round(detector.dwell_time_s * detector.sample_rate_hz)))
    std = lockin_noise_std(
        expected_total,
        n_samples,
        detector.read_noise_std_counts,
        is_dc=True,
    )
    measured = rng.normal(expected_total, std)
    if subtract_known_background:
        measured -= background
    diagnostics = {
        "samples_per_dwell": float(n_samples),
        "optical_throughput": float(optical_throughput),
        "mean_detected_counts_per_sample": float(expected_total.mean()),
    }
    return (
        measured.reshape((mixing.shape[0],) + spatial_shape),
        std.reshape((mixing.shape[0],) + spatial_shape),
        diagnostics,
    )
