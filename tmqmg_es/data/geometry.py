"""Dependency-light XYZ parsing and radius-graph construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config as C


def symbol_to_z(symbol: str) -> int:
    return C.SYMBOL_TO_Z[symbol]


def parse_xyz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return atomic numbers [N] and Cartesian positions [N, 3]."""
    lines = path.read_text().strip().splitlines()
    n_atoms = int(lines[0])
    if len(lines) < n_atoms + 2:
        raise ValueError(f"truncated XYZ file: {path}")
    atomic_numbers = np.empty(n_atoms, dtype=np.int64)
    positions = np.empty((n_atoms, 3), dtype=np.float32)
    for index, line in enumerate(lines[2 : 2 + n_atoms]):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"invalid XYZ atom line in {path}: {line!r}")
        atomic_numbers[index] = symbol_to_z(fields[0])
        positions[index] = [float(fields[1]), float(fields[2]), float(fields[3])]
    return atomic_numbers, positions


def radius_graph(positions: np.ndarray, cutoff: float) -> np.ndarray:
    """Directed edge index [2, E] with both directions for each close pair."""
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    source, destination = np.where(distances <= cutoff)
    if source.size == 0:
        nearest = np.argmin(distances, axis=1)
        source = np.arange(len(positions))
        destination = nearest
    return np.vstack([source, destination]).astype(np.int64)
