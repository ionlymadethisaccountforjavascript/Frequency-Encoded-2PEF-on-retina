"""Quantitative reconstruction and localization metrics."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def normalize_map(image: np.ndarray) -> np.ndarray:
    """Normalize one nonnegative image to the interval [0, 1].

    What happens in this function:
    1. Nonfinite values are replaced with zero.
    2. The minimum is removed to avoid negative display offsets.
    3. The map is divided by its maximum when the maximum is nonzero.
    """

    image = np.nan_to_num(np.asarray(image, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    image = image - image.min()
    maximum = float(image.max())
    return image / maximum if maximum > 0 else image


def nrmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute root-mean-square error normalized by reference RMS.

    What happens in this function:
    1. Arrays are converted to floating point and shape equality is checked.
    2. RMS error is divided by the RMS magnitude of the reference.
    3. A small denominator protects all-zero reference maps.
    """

    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have equal shapes")
    numerator = np.sqrt(np.mean((reference - estimate) ** 2))
    denominator = max(float(np.sqrt(np.mean(reference**2))), 1e-12)
    return float(numerator / denominator)


def pearson_correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Pearson correlation between flattened image intensities.

    What happens in this function:
    1. Images are flattened and mean-centred.
    2. Their dot product is divided by the product of norms.
    3. Constant images return zero instead of an undefined value.
    """

    reference = np.asarray(reference, dtype=float).ravel()
    estimate = np.asarray(estimate, dtype=float).ravel()
    reference -= reference.mean()
    estimate -= estimate.mean()
    denominator = np.linalg.norm(reference) * np.linalg.norm(estimate)
    return float(reference @ estimate / denominator) if denominator > 0 else 0.0


def ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute structural similarity after independent [0, 1] normalization.

    What happens in this function:
    1. Both images are normalized to remove arbitrary NMF scale.
    2. The data range is fixed to one.
    3. Structural similarity evaluates spatial fidelity beyond pixel-wise error.
    """

    ref = normalize_map(reference)
    est = normalize_map(estimate)
    return float(structural_similarity(ref, est, data_range=1.0))


def centroid_um(image: np.ndarray, pixel_size_um: float) -> np.ndarray:
    """Calculate the intensity-weighted centroid in physical coordinates.

    What happens in this function:
    1. Negative intensities are clipped because concentration is nonnegative.
    2. Grid indices are weighted by image intensity.
    3. Coordinates are centred and converted from pixels to micrometres.
    4. An all-zero image returns NaN coordinates.
    """

    image = np.clip(np.asarray(image, dtype=float), 0.0, None)
    total = float(image.sum())
    if total <= 0:
        return np.full(image.ndim, np.nan)
    coordinates = np.indices(image.shape, dtype=float)
    centroid_indices = np.array([(axis * image).sum() / total for axis in coordinates])
    centre_indices = (np.array(image.shape, dtype=float) - 1.0) / 2.0
    return (centroid_indices - centre_indices) * pixel_size_um


def localization_error_um(
    reference: np.ndarray,
    estimate: np.ndarray,
    pixel_size_um: float,
) -> float:
    """Return Euclidean distance between reference and estimated centroids.

    What happens in this function:
    1. Physical centroids are calculated for both images.
    2. Their coordinate difference is measured with the Euclidean norm.
    3. Failed all-zero centroids return infinity to flag reconstruction failure.
    """

    reference_centroid = centroid_um(reference, pixel_size_um)
    estimate_centroid = centroid_um(estimate, pixel_size_um)
    if not np.all(np.isfinite(reference_centroid)) or not np.all(np.isfinite(estimate_centroid)):
        return float("inf")
    return float(np.linalg.norm(reference_centroid - estimate_centroid))


def support_crosstalk(reference: np.ndarray, estimate_other: np.ndarray) -> float:
    """Measure leakage of another component into a reference component's support.

    What happens in this function:
    1. The reference support is defined by values above 20% of its maximum.
    2. Mean contaminating signal inside and outside that support is calculated.
    3. Their ratio reports how strongly the other component leaks into the region.
    4. The metric is heuristic and should be accompanied by NRMSE and correlation.
    """

    reference = normalize_map(reference)
    other = normalize_map(estimate_other)
    support = reference >= 0.2
    if not np.any(support):
        return 0.0
    inside = float(other[support].mean())
    outside = float(other[~support].mean()) if np.any(~support) else 0.0
    return float(inside / max(outside, 1e-12))


def component_metrics(
    reference_maps: np.ndarray,
    estimated_maps: np.ndarray,
    species: list[str] | tuple[str, ...],
    pixel_size_um: float,
) -> list[dict[str, float | str]]:
    """Calculate a standard metric row for every recovered species map.

    What happens in this function:
    1. Reference and estimated component counts are checked.
    2. Estimated scale is fitted by nonnegative least squares in closed form.
    3. NRMSE, correlation, SSIM, and localization error are calculated.
    4. Rows are returned as dictionaries ready for a CSV table.
    """

    reference_maps = np.asarray(reference_maps, dtype=float)
    estimated_maps = np.asarray(estimated_maps, dtype=float)
    if reference_maps.shape != estimated_maps.shape:
        raise ValueError("reference_maps and estimated_maps must have equal shapes")
    if reference_maps.shape[0] != len(species):
        raise ValueError("species names must match component count")

    rows: list[dict[str, float | str]] = []
    for index, name in enumerate(species):
        reference = reference_maps[index]
        estimate = estimated_maps[index]
        scale = float(np.sum(reference * estimate) / max(np.sum(estimate**2), 1e-12))
        scale = max(scale, 0.0)
        estimate_scaled = estimate * scale
        rows.append(
            {
                "species": name,
                "nrmse": nrmse(reference, estimate_scaled),
                "pearson": pearson_correlation(reference, estimate_scaled),
                "ssim": ssim(reference, estimate_scaled),
                "localization_error_um": localization_error_um(
                    reference,
                    estimate_scaled,
                    pixel_size_um,
                ),
                "fitted_scale": scale,
            }
        )
    return rows
