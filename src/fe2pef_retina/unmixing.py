"""Calibrated nonnegative inversion and regularized blind NNMF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment, nnls


@dataclass
class NMFResult:
    """Contain one NMF solution and its convergence diagnostics.

    What happens in this class:
    1. ``signature`` stores channel signatures for recovered components.
    2. ``concentrations`` stores the recovered component images in flattened form.
    3. ``objective`` stores the final regularized reconstruction objective.
    4. ``iterations`` records how many multiplicative updates were performed.
    """

    signature: np.ndarray
    concentrations: np.ndarray
    objective: float
    iterations: int


def calibrated_nnls_unmix(signal: np.ndarray, signature: np.ndarray) -> np.ndarray:
    """Recover nonnegative concentration maps using a known signature matrix.

    What happens in this function:
    1. Spatial dimensions are flattened while channel order is preserved.
    2. SciPy's nonnegative least-squares solver is applied independently per pixel.
    3. Negative concentrations are forbidden by the optimizer rather than clipped later.
    4. Recovered maps are reshaped to ``(species, *spatial_shape)``.
    """

    signal = np.asarray(signal, dtype=float)
    signature = np.asarray(signature, dtype=float)
    if signal.ndim < 2:
        raise ValueError("signal must have one channel axis and spatial dimensions")
    if signature.shape[0] != signal.shape[0]:
        raise ValueError("signature rows must equal the number of signal channels")
    spatial_shape = signal.shape[1:]
    flat = signal.reshape(signal.shape[0], -1)
    recovered = np.empty((signature.shape[1], flat.shape[1]), dtype=float)
    for pixel in range(flat.shape[1]):
        recovered[:, pixel], _ = nnls(signature, np.clip(flat[:, pixel], 0.0, None))
    return recovered.reshape((signature.shape[1],) + spatial_shape)


def _nmf_objective(
    signal: np.ndarray,
    signature: np.ndarray,
    concentrations: np.ndarray,
    alpha_l1: float,
    alpha_l2: float,
) -> float:
    """Evaluate the regularized NMF objective used for model selection.

    What happens in this function:
    1. Frobenius reconstruction error measures data mismatch.
    2. L1 concentration penalty promotes sparse fluorophore maps.
    3. L2 concentration penalty stabilizes ill-conditioned inversion.
    4. The three nonnegative terms are added into one scalar objective.
    """

    residual = signal - signature @ concentrations
    return float(
        np.sum(residual**2)
        + alpha_l1 * np.sum(np.abs(concentrations))
        + alpha_l2 * np.sum(concentrations**2)
    )


def regularized_nmf(
    signal: np.ndarray,
    components: int,
    alpha_l1: float,
    alpha_l2: float,
    iterations: int,
    restarts: int,
    tolerance: float,
    rng: np.random.Generator,
    initial_signature: np.ndarray | None = None,
) -> NMFResult:
    """Factor nonnegative FE channels into signatures and concentration maps.

    What happens in this function:
    1. Signal values are clipped at zero because NNMF requires nonnegative data.
    2. Random positive signatures and concentrations initialize each restart.
    3. Multiplicative updates follow the regularized equations used by Heuke et al.
    4. Signature columns are normalized after each update to control scale ambiguity.
    5. Convergence is checked through relative objective improvement.
    6. The lowest-objective restart is returned.
    """

    matrix = np.clip(np.asarray(signal, dtype=float), 0.0, None)
    if matrix.ndim != 2:
        raise ValueError("signal must have shape (channels, pixels)")
    if components <= 0 or components > min(matrix.shape):
        raise ValueError("components must be positive and no larger than matrix rank limits")
    if min(alpha_l1, alpha_l2, tolerance) < 0:
        raise ValueError("regularization and tolerance values must be nonnegative")
    if iterations <= 0 or restarts <= 0:
        raise ValueError("iterations and restarts must be positive")

    epsilon = 1e-12
    best: NMFResult | None = None
    for restart in range(restarts):
        if initial_signature is not None and restart == 0:
            signature = np.clip(np.asarray(initial_signature, dtype=float), epsilon, None).copy()
            if signature.shape != (matrix.shape[0], components):
                raise ValueError("initial_signature has incompatible shape")
        else:
            signature = rng.random((matrix.shape[0], components)) + 0.1
        concentrations = rng.random((components, matrix.shape[1])) + 0.1
        previous_objective = np.inf

        performed = iterations
        for iteration in range(iterations):
            signature *= (matrix @ concentrations.T) / (
                signature @ concentrations @ concentrations.T + epsilon
            )
            column_sums = signature.sum(axis=0, keepdims=True)
            column_sums = np.where(column_sums > epsilon, column_sums, 1.0)
            signature /= column_sums
            concentrations *= column_sums.T

            numerator = signature.T @ matrix
            denominator = (
                signature.T @ signature @ concentrations
                + alpha_l1
                + alpha_l2 * concentrations
                + epsilon
            )
            concentrations *= numerator / denominator

            if iteration % 10 == 0 or iteration == iterations - 1:
                objective = _nmf_objective(
                    matrix,
                    signature,
                    concentrations,
                    alpha_l1,
                    alpha_l2,
                )
                relative_change = abs(previous_objective - objective) / max(
                    abs(previous_objective), 1.0
                )
                if relative_change < tolerance:
                    performed = iteration + 1
                    break
                previous_objective = objective

        objective = _nmf_objective(
            matrix,
            signature,
            concentrations,
            alpha_l1,
            alpha_l2,
        )
        candidate = NMFResult(signature, concentrations, objective, performed)
        if best is None or candidate.objective < best.objective:
            best = candidate

    if best is None:
        raise RuntimeError("NMF did not produce a solution")
    return best


def align_components(
    recovered_maps: np.ndarray,
    reference_maps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align arbitrary NMF component order to known reference maps for evaluation.

    What happens in this function:
    1. Recovered and reference maps are flattened and mean-centred.
    2. Absolute correlation becomes the component-assignment score.
    3. The Hungarian algorithm finds the maximum-total-correlation permutation.
    4. Reordered maps and the permutation indices are returned.
    5. This alignment is only for benchmarking; real blind data lack ground truth.
    """

    recovered = np.asarray(recovered_maps, dtype=float)
    reference = np.asarray(reference_maps, dtype=float)
    if recovered.shape[0] != reference.shape[0]:
        raise ValueError("recovered and reference maps must have the same component count")
    rec_flat = recovered.reshape(recovered.shape[0], -1)
    ref_flat = reference.reshape(reference.shape[0], -1)
    rec_flat = rec_flat - rec_flat.mean(axis=1, keepdims=True)
    ref_flat = ref_flat - ref_flat.mean(axis=1, keepdims=True)
    rec_norm = np.linalg.norm(rec_flat, axis=1, keepdims=True)
    ref_norm = np.linalg.norm(ref_flat, axis=1, keepdims=True)
    correlation = (rec_flat @ ref_flat.T) / np.maximum(rec_norm @ ref_norm.T, 1e-12)
    row_ind, col_ind = linear_sum_assignment(-np.abs(correlation))
    order = np.empty(recovered.shape[0], dtype=int)
    for recovered_index, reference_index in zip(row_ind, col_ind):
        order[reference_index] = recovered_index
    return recovered[order], order


def reshape_nmf_concentrations(result: NMFResult, spatial_shape: tuple[int, ...]) -> np.ndarray:
    """Reshape flattened NMF concentration vectors into spatial maps.

    What happens in this function:
    1. The expected pixel count is calculated from the requested spatial shape.
    2. The NMF concentration matrix is checked against that pixel count.
    3. Components are reshaped while preserving their first-axis order.
    """

    expected_pixels = int(np.prod(spatial_shape))
    if result.concentrations.shape[1] != expected_pixels:
        raise ValueError("NMF pixel count does not match spatial_shape")
    return result.concentrations.reshape((result.concentrations.shape[0],) + spatial_shape)
