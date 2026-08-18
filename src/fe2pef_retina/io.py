"""Reproducible output writing and provenance helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    """Create an output directory and return it as a Path.

    What happens in this function:
    1. The supplied path is converted to ``Path``.
    2. Missing parent directories are created.
    3. Existing directories are accepted without error.
    """

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _json_ready(value: Any) -> Any:
    """Convert dataclasses, arrays, paths, and tuples into JSON-compatible values.

    What happens in this function:
    1. Dataclasses are expanded into dictionaries.
    2. Paths and NumPy scalar types are converted to plain Python values.
    3. Containers are recursively traversed.
    4. Remaining scalar values pass through unchanged.
    """

    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def save_json(path: str | Path, payload: Any) -> None:
    """Write a human-readable JSON file with stable formatting.

    What happens in this function:
    1. Parent directories are created if needed.
    2. The payload is converted into JSON-compatible values.
    3. UTF-8 text is written with indentation and sorted keys.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)


def save_array_bundle(path: str | Path, **arrays: np.ndarray) -> None:
    """Save named NumPy arrays in a compressed NPZ archive.

    What happens in this function:
    1. Parent directories are created.
    2. Every value is converted to a NumPy array.
    3. Compressed storage preserves exact numeric outputs for reproducibility.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{name: np.asarray(value) for name, value in arrays.items()})


def save_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of metric or sweep rows to CSV.

    What happens in this function:
    1. Parent directories are created.
    2. Dictionaries become a pandas DataFrame.
    3. Rows are written without an extra index column.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
