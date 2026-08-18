"""Synthetic retinal concentration maps and user-data loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter


def _coordinate_grid(shape: Sequence[int], pixel_size_um: float) -> tuple[np.ndarray, ...]:
    """Create centred physical-coordinate arrays for a spatial grid.

    What happens in this function:
    1. Each array index is shifted so the field centre is zero.
    2. Pixel spacing converts coordinates into micrometres.
    3. ``meshgrid`` returns one coordinate array per spatial dimension.
    """

    axes = [
        (np.arange(length, dtype=float) - (length - 1) / 2.0) * pixel_size_um
        for length in shape
    ]
    return tuple(np.meshgrid(*axes, indexing="ij"))


def gaussian_spot(
    shape: Sequence[int],
    pixel_size_um: float,
    centre_um: Sequence[float],
    sigma_um: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate an N-dimensional Gaussian concentration spot.

    What happens in this function:
    1. A physical coordinate grid is created in micrometres.
    2. Squared distance from the requested centre is accumulated.
    3. A Gaussian profile is evaluated and scaled by amplitude.
    """

    if len(shape) != len(centre_um):
        raise ValueError("centre_um must have one coordinate per spatial dimension")
    if sigma_um <= 0 or pixel_size_um <= 0 or amplitude < 0:
        raise ValueError("sigma and pixel size must be positive; amplitude nonnegative")
    coordinates = _coordinate_grid(shape, pixel_size_um)
    radius_squared = np.zeros(shape, dtype=float)
    for coordinate, centre in zip(coordinates, centre_um):
        radius_squared += (coordinate - centre) ** 2
    return amplitude * np.exp(-0.5 * radius_squared / sigma_um**2)


def two_spot_phantom(
    shape: Sequence[int],
    pixel_size_um: float,
    separation_um: float,
    sigma_um: float = 1.2,
) -> np.ndarray:
    """Create two species represented by separated Gaussian spots.

    What happens in this function:
    1. Two centres are placed symmetrically around the field centre.
    2. Each centre becomes one species-specific Gaussian concentration map.
    3. The output has shape ``(2, *shape)`` for direct use by the forward model.
    """

    if len(shape) != 2:
        raise ValueError("two_spot_phantom currently expects a 2-D shape")
    centre1 = (0.0, -separation_um / 2.0)
    centre2 = (0.0, separation_um / 2.0)
    first = gaussian_spot(shape, pixel_size_um, centre1, sigma_um, amplitude=1.0)
    second = gaussian_spot(shape, pixel_size_um, centre2, sigma_um, amplitude=1.0)
    return np.stack([first, second], axis=0)


def overlap_phantom(
    shape: Sequence[int],
    pixel_size_um: float,
    overlap_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create two spatially overlapping but non-identical concentration maps.

    What happens in this function:
    1. Smooth random fields create a shared biological-looking texture.
    2. Independent smooth fields create species-specific structure.
    3. ``overlap_fraction`` mixes shared and independent components.
    4. Each output map is normalized to a maximum of one.
    """

    if not 0 <= overlap_fraction <= 1:
        raise ValueError("overlap_fraction must be between zero and one")
    if len(shape) != 2:
        raise ValueError("overlap_phantom currently expects a 2-D shape")
    sigma_px = max(1.0, 4.0 / pixel_size_um)
    shared = gaussian_filter(rng.random(shape), sigma=sigma_px)
    unique1 = gaussian_filter(rng.random(shape), sigma=sigma_px)
    unique2 = gaussian_filter(rng.random(shape), sigma=sigma_px)
    first = overlap_fraction * shared + (1.0 - overlap_fraction) * unique1
    second = overlap_fraction * shared + (1.0 - overlap_fraction) * unique2
    first /= max(float(first.max()), 1e-12)
    second /= max(float(second.max()), 1e-12)
    return np.stack([first, second], axis=0)


def rpe_mosaic_phantom(
    shape: Sequence[int],
    pixel_size_um: float,
    n_cells: int,
    granules_per_cell: int,
    overlap_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an RPE-like cell mosaic with granular and diffuse species maps.

    What happens in this function:
    1. Random seed points define Voronoi-like polygonal RPE cells.
    2. A2E-like signal is represented by intracellular granular deposits.
    3. FAD-like signal is represented by diffuse, cell-varying cytoplasmic signal.
    4. Shared morphology introduces configurable spatial overlap.
    5. The generator is a morphology prior, not measured molecular ground truth.
    """

    if len(shape) != 2:
        raise ValueError("rpe_mosaic_phantom expects a 2-D shape")
    if n_cells < 2 or granules_per_cell < 1:
        raise ValueError("n_cells and granules_per_cell must be positive")
    if not 0 <= overlap_fraction <= 1:
        raise ValueError("overlap_fraction must be between zero and one")

    rows, cols = shape
    seed_rows = rng.uniform(0, rows - 1, size=n_cells)
    seed_cols = rng.uniform(0, cols - 1, size=n_cells)
    rr, cc = np.indices(shape)
    distances = (
        (rr[..., None] - seed_rows[None, None, :]) ** 2
        + (cc[..., None] - seed_cols[None, None, :]) ** 2
    )
    labels = np.argmin(distances, axis=-1)

    a2e_like = np.zeros(shape, dtype=float)
    fad_like = np.zeros(shape, dtype=float)
    shared = np.zeros(shape, dtype=float)

    for cell_id in range(n_cells):
        mask = labels == cell_id
        coordinates = np.argwhere(mask)
        if coordinates.size == 0:
            continue
        cell_brightness = rng.lognormal(mean=0.0, sigma=0.25)
        fad_like[mask] = cell_brightness
        shared[mask] = rng.lognormal(mean=-0.1, sigma=0.2)

        chosen = coordinates[
            rng.integers(0, len(coordinates), size=granules_per_cell)
        ]
        for row, col in chosen:
            sigma_px = rng.uniform(0.6, 1.5) / pixel_size_um
            amplitude = rng.lognormal(mean=0.0, sigma=0.45)
            impulse = np.zeros(shape, dtype=float)
            impulse[int(row), int(col)] = amplitude
            a2e_like += gaussian_filter(impulse, sigma=max(0.45, sigma_px))

    boundary = np.zeros(shape, dtype=bool)
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    cell_interior = gaussian_filter((~boundary).astype(float), sigma=0.8)
    fad_like *= cell_interior
    shared *= cell_interior

    shared /= max(float(shared.max()), 1e-12)
    a2e_like /= max(float(a2e_like.max()), 1e-12)
    fad_like /= max(float(fad_like.max()), 1e-12)
    first = (1.0 - overlap_fraction) * a2e_like + overlap_fraction * shared
    second = (1.0 - overlap_fraction) * fad_like + overlap_fraction * shared
    first /= max(float(first.max()), 1e-12)
    second /= max(float(second.max()), 1e-12)
    return np.stack([first, second], axis=0), labels


def rpe_volume_phantom(
    shape: Sequence[int],
    pixel_size_um: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a small 3-D two-species retinal proof-of-concept volume.

    What happens in this function:
    1. A 2-D RPE mosaic is created as the lateral morphology.
    2. Species are assigned different axial Gaussian profiles.
    3. The maps are normalized and returned as ``(species, z, y, x)``.
    4. This volume demonstrates software scaling, not a validated retinal atlas.
    """

    if len(shape) != 3:
        raise ValueError("rpe_volume_phantom expects shape (z, y, x)")
    z_size, y_size, x_size = shape
    lateral, _ = rpe_mosaic_phantom(
        (y_size, x_size),
        pixel_size_um,
        n_cells=max(8, (y_size * x_size) // 700),
        granules_per_cell=10,
        overlap_fraction=0.25,
        rng=rng,
    )
    z = (np.arange(z_size) - (z_size - 1) / 2.0) * pixel_size_um
    profile1 = np.exp(-0.5 * ((z + 1.2) / 1.5) ** 2)
    profile2 = np.exp(-0.5 * ((z - 1.0) / 2.2) ** 2)
    first = profile1[:, None, None] * lateral[0][None, :, :]
    second = profile2[:, None, None] * lateral[1][None, :, :]
    return np.stack([first, second], axis=0)


def load_concentration_maps(path: str | Path) -> np.ndarray:
    """Load user-supplied concentration maps from NPY or NPZ files.

    What happens in this function:
    1. NPY arrays are loaded directly.
    2. NPZ archives must contain a ``concentration_maps`` array.
    3. The first dimension is interpreted as species.
    4. Values are checked for finite nonnegative concentrations.
    """

    path = Path(path)
    if path.suffix.lower() == ".npy":
        maps = np.load(path)
    elif path.suffix.lower() == ".npz":
        archive = np.load(path)
        if "concentration_maps" not in archive:
            raise ValueError("NPZ file must contain 'concentration_maps'")
        maps = archive["concentration_maps"]
    else:
        raise ValueError("supported map formats are .npy and .npz")
    maps = np.asarray(maps, dtype=float)
    if maps.ndim not in (3, 4) or maps.shape[0] < 1:
        raise ValueError("maps must have shape (species, y, x) or (species, z, y, x)")
    if not np.all(np.isfinite(maps)) or np.any(maps < 0):
        raise ValueError("concentration maps must be finite and nonnegative")
    return maps
