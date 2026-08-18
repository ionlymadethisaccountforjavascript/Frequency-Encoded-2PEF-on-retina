"""Normalize user-provided species maps into the package NPZ convention."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def normalize_species_maps(maps: np.ndarray) -> np.ndarray:
    """Normalize each nonnegative species map independently to a maximum of one.

    What happens in this function:
    1. The first array axis is interpreted as fluorophore species.
    2. Nonfinite or negative values are rejected.
    3. Each species is divided by its own maximum when nonzero.
    4. Relative concentration patterns are retained while absolute units are discarded.
    """

    maps = np.asarray(maps, dtype=float)
    if maps.ndim not in (3, 4):
        raise ValueError("expected (species,y,x) or (species,z,y,x)")
    if not np.all(np.isfinite(maps)) or np.any(maps < 0):
        raise ValueError("maps must be finite and nonnegative")
    axes = tuple(range(1, maps.ndim))
    maxima = maps.max(axis=axes, keepdims=True)
    maxima = np.where(maxima > 0, maxima, 1.0)
    return maps / maxima


def main(argv: list[str] | None = None) -> int:
    """Read NPY maps, normalize them, and write a standard NPZ archive.

    What happens in this function:
    1. Input and output file arguments are parsed.
    2. The input array is loaded from NPY format.
    3. Species maps are validated and normalized.
    4. The output archive uses the key ``concentration_maps``.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args(argv)
    maps = normalize_species_maps(np.load(arguments.input))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output, concentration_maps=maps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
