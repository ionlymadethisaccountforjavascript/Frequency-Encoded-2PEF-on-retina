"""Fluorophore pathway responses and frequency-channel signatures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import LaserConfig

REQUIRED_RESPONSE_COLUMNS = {
    "species",
    "sigma_11",
    "sigma_12",
    "sigma_22",
    "brightness",
    "provenance",
}


def load_pathway_responses(path: str | Path, species: Iterable[str]) -> pd.DataFrame:
    """Load normalized two-colour pathway responses for selected species.

    What happens in this function:
    1. A CSV table is read and required columns are checked.
    2. Requested species are selected in the exact requested order.
    3. Nonnegative and finite response values are enforced.
    4. Provenance remains attached so illustrative values cannot be mistaken for data.
    """

    frame = pd.read_csv(path)
    missing_columns = REQUIRED_RESPONSE_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"response table is missing columns: {sorted(missing_columns)}")

    requested = list(species)
    indexed = frame.set_index("species", drop=False)
    missing_species = [name for name in requested if name not in indexed.index]
    if missing_species:
        raise ValueError(f"response table is missing species: {missing_species}")
    selected = indexed.loc[requested].reset_index(drop=True)

    numeric_columns = ["sigma_11", "sigma_12", "sigma_22", "brightness"]
    values = selected[numeric_columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("pathway responses and brightness must be finite and nonnegative")
    if np.any(values[:, :3].sum(axis=1) <= 0):
        raise ValueError("each species must respond to at least one excitation pathway")
    return selected


def pathway_response_matrix(response_frame: pd.DataFrame) -> np.ndarray:
    """Convert a response table into a pathways-by-species matrix.

    What happens in this function:
    1. Columns sigma_11, sigma_12, and sigma_22 are extracted.
    2. Species brightness multiplies every pathway for that species.
    3. The returned matrix has rows [11, 12, 22] and columns as species.
    """

    responses = response_frame[["sigma_11", "sigma_12", "sigma_22"]].to_numpy(float).T
    brightness = response_frame["brightness"].to_numpy(float)[None, :]
    return responses * brightness


def channel_frequency_hz(channel: str, laser1: LaserConfig, laser2: LaserConfig) -> float:
    """Map a named lock-in channel to its physical modulation frequency.

    What happens in this function:
    1. Fundamental, harmonic, sum, and difference names are recognized.
    2. The corresponding nonnegative frequency is calculated.
    3. DC maps to zero hertz.
    """

    f1 = laser1.modulation_frequency_hz
    f2 = laser2.modulation_frequency_hz
    mapping = {
        "dc": 0.0,
        "f1": f1,
        "f2": f2,
        "difference": abs(f1 - f2),
        "sum": f1 + f2,
        "2f1": 2.0 * f1,
        "2f2": 2.0 * f2,
    }
    try:
        return float(mapping[channel])
    except KeyError as exc:
        raise ValueError(f"unknown channel: {channel}") from exc


def build_frequency_signature_matrix(
    response_frame: pd.DataFrame,
    laser1: LaserConfig,
    laser2: LaserConfig,
    temporal_overlap: float,
    channels: Iterable[str],
) -> np.ndarray:
    """Build the analytic FE-2PEF channel-by-species mixing matrix.

    What happens in this function:
    1. Each species contributes to I1^2, I1*I2, and I2^2 pathways.
    2. Sinusoidal laser modulation is expanded into DC and sideband terms.
    3. Laser powers, modulation depths, and pulse overlap scale each coefficient.
    4. Rows follow the requested lock-in channels and columns follow species order.
    5. Signs are chosen to match the phase-aware references in ``lockin.py``.
    """

    if not 0 <= temporal_overlap <= 1:
        raise ValueError("temporal_overlap must be between zero and one")

    pathway = pathway_response_matrix(response_frame)
    sigma11, sigma12, sigma22 = pathway
    p1 = laser1.average_power_mw
    p2 = laser2.average_power_mw
    m1 = laser1.modulation_depth
    m2 = laser2.modulation_depth
    rho = temporal_overlap

    coefficients = {
        "dc": (
            sigma11 * p1**2 * (1.0 + 0.5 * m1**2)
            + sigma22 * p2**2 * (1.0 + 0.5 * m2**2)
            + 2.0 * rho * sigma12 * p1 * p2
        ),
        "f1": 2.0 * m1 * (sigma11 * p1**2 + rho * sigma12 * p1 * p2),
        "f2": 2.0 * m2 * (sigma22 * p2**2 + rho * sigma12 * p1 * p2),
        "difference": rho * sigma12 * p1 * p2 * m1 * m2,
        "sum": rho * sigma12 * p1 * p2 * m1 * m2,
        "2f1": 0.5 * sigma11 * p1**2 * m1**2,
        "2f2": 0.5 * sigma22 * p2**2 * m2**2,
    }
    return np.vstack([coefficients[channel] for channel in channels]).astype(float)


def condition_number(matrix: np.ndarray) -> float:
    """Return the 2-norm condition number of a mixing matrix.

    What happens in this function:
    1. The matrix is converted to floating point.
    2. Singular values are used through NumPy's condition-number routine.
    3. Large values flag fluorophore signatures that are hard to distinguish.
    """

    matrix = np.asarray(matrix, dtype=float)
    return float(np.linalg.cond(matrix))


def normalize_signature_columns(matrix: np.ndarray) -> np.ndarray:
    """Normalize each fluorophore signature to unit Euclidean norm.

    What happens in this function:
    1. A norm is calculated for every species column.
    2. Zero columns are rejected because they cannot be unmixed.
    3. Normalization removes arbitrary brightness scale for geometry comparisons.
    """

    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("signature matrix contains a zero column")
    return matrix / norms
