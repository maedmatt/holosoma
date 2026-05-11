"""Accumulate named per-step signals and save them to NPZ.

Auto-discovers field names from the first append; later appends must use
the same keys (typos and missing fields raise immediately rather than
producing a quietly truncated dump). Scalars, numpy arrays, and torch
tensors are all accepted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


class TimeseriesLogger:
    """Append rows by keyword, save them as a single NPZ with one array per field."""

    def __init__(self) -> None:
        self._bufs: dict[str, list[np.ndarray]] = {}
        self._keys: tuple[str, ...] | None = None

    def append(self, **kwargs: Any) -> None:
        """Append one row. Keys must match the first append; mismatch raises."""
        if self._keys is None:
            self._keys = tuple(kwargs.keys())
            for k in self._keys:
                self._bufs[k] = []
        elif tuple(kwargs.keys()) != self._keys:
            raise ValueError(f"key mismatch: expected {self._keys}, got {tuple(kwargs.keys())}")
        for k, v in kwargs.items():
            self._bufs[k].append(_to_numpy(v))

    def save(self, path: str | Path) -> int:
        """Save to NPZ, returns row count."""
        arrays = {k: np.stack(v) for k, v in self._bufs.items()}
        n_rows = next(iter(arrays.values())).shape[0]
        np.savez(path, **arrays)
        return n_rows

    def clear(self) -> None:
        self._bufs.clear()
        self._keys = None

    def __len__(self) -> int:
        if not self._bufs:
            return 0
        return len(next(iter(self._bufs.values())))
