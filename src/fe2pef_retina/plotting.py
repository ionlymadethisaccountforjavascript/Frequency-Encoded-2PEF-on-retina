"""Plotting helpers for manuscript-style simulation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import normalize_map


def _display_projection(image: np.ndarray) -> np.ndarray:
    """Convert a scalar 2-D or 3-D field into a displayable 2-D image.

    What happens in this function:
    1. A two-dimensional input is returned without modification.
    2. A three-dimensional volume is maximum-intensity projected along its first axis.
    3. Other dimensionalities are rejected so plotting errors are explicit.
    4. The numerical arrays saved by experiments remain unchanged; only the figure is projected.
    """

    array = np.asarray(image, dtype=float)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return np.max(array, axis=0)
    raise ValueError("scalar images must be two-dimensional or three-dimensional")


def save_image_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    path: str | Path,
    columns: int = 3,
    colorbar: bool = True,
) -> None:
    """Save a compact grid of independently scaled scalar images.

    What happens in this function:
    1. The number of rows is calculated from image count and requested columns.
    2. Each image is normalized for morphology-focused visual comparison.
    3. Unused axes are hidden and optional color bars are added.
    4. The figure is saved with tight bounding and then closed.
    """

    if len(images) != len(titles) or len(images) == 0:
        raise ValueError("images and titles must be nonempty and have equal length")
    columns = max(1, int(columns))
    rows = int(np.ceil(len(images) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.8 * rows), squeeze=False)
    for axis, image, title in zip(axes.ravel(), images, titles):
        artist = axis.imshow(normalize_map(_display_projection(image)), origin="lower")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        if colorbar:
            figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes.ravel()[len(images):]:
        axis.axis("off")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_signature_heatmap(
    matrix: np.ndarray,
    channels: Sequence[str],
    species: Sequence[str],
    path: str | Path,
) -> None:
    """Save a labelled heatmap of a channel-by-species signature matrix.

    What happens in this function:
    1. Matrix values are displayed without column normalization.
    2. Channel and species labels are placed on the axes.
    3. Numeric values are printed in every cell for auditability.
    4. A color bar and tight layout are added before saving.
    """

    matrix = np.asarray(matrix, dtype=float)
    figure, axis = plt.subplots(figsize=(1.7 * len(species) + 2.5, 0.8 * len(channels) + 2.5))
    artist = axis.imshow(matrix, aspect="auto")
    axis.set_xticks(range(len(species)), species, rotation=30, ha="right")
    axis.set_yticks(range(len(channels)), channels)
    axis.set_xlabel("Species")
    axis.set_ylabel("Lock-in channel")
    axis.set_title("Effective FE-2PEF signature matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.3g}", ha="center", va="center")
    figure.colorbar(artist, ax=axis)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_metric_bar_chart(metrics: pd.DataFrame, path: str | Path) -> None:
    """Save a method comparison bar chart using mean species NRMSE.

    What happens in this function:
    1. Metric rows are grouped by reconstruction method.
    2. Mean NRMSE and standard deviation across species are calculated.
    3. A single bar chart summarizes lower-is-better reconstruction error.
    4. The underlying values remain available in the companion CSV file.
    """

    summary = metrics.groupby("method")["nrmse"].agg(["mean", "std"]).reset_index()
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(summary["method"], summary["mean"], yerr=summary["std"].fillna(0.0), capsize=4)
    axis.set_ylabel("Mean NRMSE (lower is better)")
    axis.set_title("Reconstruction comparison")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_sweep_plot(
    table: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    path: str | Path,
    x_label: str,
    y_label: str,
) -> None:
    """Save one line plot for a robustness sweep with replicate uncertainty.

    What happens in this function:
    1. Replicate rows are grouped by method and x-axis value.
    2. Mean and standard deviation of the requested metric are calculated.
    3. Each method is drawn as a separate line with an uncertainty band.
    4. The plot is saved and the figure is closed to avoid memory leakage.
    """

    summary = table.groupby([group, x])[y].agg(["mean", "std"]).reset_index()
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for label, subset in summary.groupby(group):
        subset = subset.sort_values(x)
        axis.plot(subset[x], subset["mean"], marker="o", label=str(label))
        standard = subset["std"].fillna(0.0).to_numpy()
        axis.fill_between(
            subset[x].to_numpy(),
            subset["mean"].to_numpy() - standard,
            subset["mean"].to_numpy() + standard,
            alpha=0.2,
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.legend()
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
