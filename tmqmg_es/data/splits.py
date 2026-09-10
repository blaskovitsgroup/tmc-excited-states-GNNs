"""Deterministic, identity-aware train/validation/locked-test split.

groups chemically identical and geometry-duplicate records before one
frozen StratifiedGroupKFold assignment. No split rerolling is supported.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from .geometry import parse_xyz

SPLIT_SEED = 20260724
N_FOLDS = 10
GEOMETRY_RMSD_THRESHOLD_ANGSTROM = 0.10
GEOMETRY_CANDIDATE_DISTANCE_ANGSTROM = 0.20


class UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            nxt = self.parent[item]
            self.parent[item] = root
            item = nxt
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        # The lexicographically smallest molecule ID is the stable component ID.
        if a > b:
            a, b = b, a
        self.parent[b] = a


def _charge_key(value: object) -> str:
    charge = float(value)
    return str(int(charge)) if charge.is_integer() else format(charge, ".12g")


def _normalized_raw_smiles(smiles: str) -> str:
    return re.sub(r"\s+", "", smiles.strip())


def canonical_identity_key(smiles: object, charge: object) -> tuple[str, str] | None:
    """Return a stable chemical identity key; missing SMILES stays ungrouped."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    charge_key = _charge_key(charge)
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            return f"canonical:{canonical}", charge_key
    except Exception:
        pass
    return f"raw:{_normalized_raw_smiles(smiles)}", charge_key


def _composition_key(z: np.ndarray) -> tuple[tuple[int, int], ...]:
    values, counts = np.unique(z, return_counts=True)
    return tuple((int(value), int(count)) for value, count in zip(values, counts))


def _sorted_pair_distances(pos: np.ndarray) -> np.ndarray:
    if len(pos) < 2:
        return np.empty(0, dtype=np.float32)
    distances = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    return np.sort(distances[np.triu_indices(len(pos), 1)]).astype(np.float32)


def _exact_geometry_fingerprint(z: np.ndarray, pos: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(repr(_composition_key(z)).encode("ascii"))
    digest.update(np.round(_sorted_pair_distances(pos), 3).tobytes())
    return digest.hexdigest()


def _atom_environment(z: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Permutation-invariant distance environment for deterministic atom matching."""
    unique_z = np.unique(z)
    rows = []
    distances = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    for atom in range(len(z)):
        features = []
        for element in unique_z:
            values = np.sort(distances[atom, z == element])
            features.extend(values.tolist())
        rows.append(features)
    return np.asarray(rows, dtype=float)


def _element_preserving_assignment(
    z_ref: np.ndarray,
    pos_ref: np.ndarray,
    z_query: np.ndarray,
    pos_query: np.ndarray,
) -> np.ndarray | None:
    if _composition_key(z_ref) != _composition_key(z_query):
        return None
    from scipy.optimize import linear_sum_assignment

    env_ref = _atom_environment(z_ref, pos_ref)
    env_query = _atom_environment(z_query, pos_query)
    assignment = np.full(len(z_ref), -1, dtype=int)
    for element in np.unique(z_ref):
        ref_idx = np.flatnonzero(z_ref == element)
        query_idx = np.flatnonzero(z_query == element)
        cost = np.linalg.norm(
            env_ref[ref_idx, None, :] - env_query[None, query_idx, :], axis=-1
        )
        rows, cols = linear_sum_assignment(cost)
        assignment[ref_idx[rows]] = query_idx[cols]
    return assignment if np.all(assignment >= 0) else None


def kabsch_rmsd(
    z_ref: np.ndarray,
    pos_ref: np.ndarray,
    z_query: np.ndarray,
    pos_query: np.ndarray,
) -> float:
    """Element-matched proper-rotation RMSD after centering."""
    assignment = _element_preserving_assignment(z_ref, pos_ref, z_query, pos_query)
    if assignment is None:
        return float("inf")
    reference = np.asarray(pos_ref, dtype=float)
    query = np.asarray(pos_query[assignment], dtype=float)
    reference -= reference.mean(axis=0)
    query -= query.mean(axis=0)
    covariance = query.T @ reference
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    aligned = query @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=1))))


@dataclass
class GeometryRecord:
    molecule_id: str
    z: np.ndarray
    pos: np.ndarray
    distances: np.ndarray
    composition: tuple[tuple[int, int], ...]
    exact_fingerprint: str


def _load_geometry_records(ids: list[str], xyz_dir: Path) -> list[GeometryRecord]:
    records = []
    for molecule_id in ids:
        path = xyz_dir / f"{molecule_id}.xyz"
        if not path.exists():
            continue
        z, pos = parse_xyz(path)
        records.append(
            GeometryRecord(
                molecule_id=molecule_id,
                z=z,
                pos=pos,
                distances=_sorted_pair_distances(pos),
                composition=_composition_key(z),
                exact_fingerprint=_exact_geometry_fingerprint(z, pos),
            )
        )
    return records


def _near_geometry_pairs(
    records: list[GeometryRecord],
    rmsd_threshold: float,
) -> list[tuple[str, str, float]]:
    """Generate distance-spectrum candidates, then verify by Kabsch RMSD."""
    from sklearn.neighbors import NearestNeighbors

    by_shape: dict[tuple, list[GeometryRecord]] = defaultdict(list)
    for record in records:
        by_shape[(record.composition, len(record.distances))].append(record)

    verified: list[tuple[str, str, float]] = []
    for group in by_shape.values():
        if len(group) < 2:
            continue
        matrix = np.stack([record.distances for record in group])
        neighbors = NearestNeighbors(
            radius=GEOMETRY_CANDIDATE_DISTANCE_ANGSTROM,
            metric="chebyshev",
            algorithm="brute",
            n_jobs=-1,
        ).fit(matrix)
        candidate_lists = neighbors.radius_neighbors(
            matrix, return_distance=False, sort_results=False
        )
        for left_idx, candidates in enumerate(candidate_lists):
            left = group[left_idx]
            for right_idx in candidates:
                if right_idx <= left_idx:
                    continue
                right = group[int(right_idx)]
                rmsd = kabsch_rmsd(left.z, left.pos, right.z, right.pos)
                if rmsd <= rmsd_threshold:
                    verified.append((left.molecule_id, right.molecule_id, rmsd))
    return verified


def build_identity_groups(
    df: pd.DataFrame,
    xyz_dir: Path = C.XYZ_DIR,
    rmsd_threshold: float = GEOMETRY_RMSD_THRESHOLD_ANGSTROM,
) -> tuple[pd.DataFrame, dict]:
    """Return molecule-to-component mapping and an auditable grouping summary."""
    ordered = df.sort_values("id", kind="stable").reset_index(drop=True)
    ids = ordered["id"].astype(str).tolist()
    union = UnionFind(ids)

    chemical_buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    key_by_id: dict[str, str | None] = {}
    for row in ordered.itertuples(index=False):
        key = canonical_identity_key(row.smiles, row.charge)
        key_by_id[row.id] = None if key is None else "|".join(key)
        if key is not None:
            chemical_buckets[key].append(row.id)
    chemical_links = 0
    for members in chemical_buckets.values():
        first = members[0]
        for other in members[1:]:
            union.union(first, other)
            chemical_links += 1

    geometries = _load_geometry_records(ids, Path(xyz_dir))
    exact_buckets: dict[str, list[str]] = defaultdict(list)
    for record in geometries:
        exact_buckets[record.exact_fingerprint].append(record.molecule_id)
    exact_links = 0
    for members in exact_buckets.values():
        first = members[0]
        for other in members[1:]:
            union.union(first, other)
            exact_links += 1

    near_pairs = _near_geometry_pairs(geometries, rmsd_threshold)
    for left, right, _ in near_pairs:
        union.union(left, right)

    groups = pd.DataFrame(
        {
            "id": ids,
            "identity_group": [union.find(molecule_id) for molecule_id in ids],
            "chemical_identity_key": [key_by_id[molecule_id] for molecule_id in ids],
        }
    )
    sizes = groups["identity_group"].value_counts()
    summary = {
        "n_molecules": len(groups),
        "n_identity_groups": int(sizes.size),
        "n_non_singleton_groups": int((sizes > 1).sum()),
        "largest_group": int(sizes.max()),
        "chemical_union_links": chemical_links,
        "exact_geometry_union_links": exact_links,
        "near_geometry_verified_pairs": len(near_pairs),
        "near_geometry_rmsd_threshold_angstrom": rmsd_threshold,
        "near_geometry_pairs": [
            {"left": left, "right": right, "rmsd_angstrom": rmsd}
            for left, right, rmsd in near_pairs
        ],
    }
    return groups, summary


def _stratification_labels(df: pd.DataFrame) -> pd.Series:
    ranks = df["n_atoms"].rank(method="first")
    atom_bin = pd.qcut(ranks, q=5, labels=False).astype(int)
    detailed = (
        df["metal_center"].astype(str)
        + "|q="
        + df["charge"].map(_charge_key)
        + "|size="
        + atom_bin.astype(str)
        + "|gvis="
        + df["mask_peak_vis_gasphase"].astype(int).astype(str)
        + "|avis="
        + df["mask_peak_vis_acetone"].astype(int).astype(str)
    )
    counts = detailed.value_counts()
    rare = detailed.map(counts).lt(N_FOLDS)
    if rare.any():
        detailed = detailed.mask(rare, "pooled_rare_composite_strata")
    if detailed.value_counts().min() < N_FOLDS:
        raise RuntimeError("stratification contains fewer rows than folds")
    return detailed


def build_confirmatory_split(
    df: pd.DataFrame,
    xyz_dir: Path = C.XYZ_DIR,
    seed: int = SPLIT_SEED,
) -> tuple[dict, pd.DataFrame, dict]:
    """Build the one permitted split; no retry or reroll path exists."""
    if seed != SPLIT_SEED:
        raise ValueError(
            f"confirmatory split seed is frozen at {SPLIT_SEED}, received {seed}"
        )
    from sklearn.model_selection import StratifiedGroupKFold

    ordered = df.sort_values("id", kind="stable").reset_index(drop=True)
    identity, identity_summary = build_identity_groups(ordered, Path(xyz_dir))
    if identity["id"].tolist() != ordered["id"].tolist():
        raise RuntimeError("identity rows do not align with sorted input rows")

    y = _stratification_labels(ordered)
    groups = identity["identity_group"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold = np.full(len(ordered), -1, dtype=int)
    dummy = np.zeros((len(ordered), 1), dtype=np.int8)
    for fold_index, (_, held_index) in enumerate(
        splitter.split(dummy, y=y, groups=groups)
    ):
        fold[held_index] = fold_index
    if np.any(fold < 0):
        raise RuntimeError("at least one molecule was not assigned to a fold")

    identity = identity.copy()
    identity["fold"] = fold
    identity["partition"] = np.where(
        fold == 0, "test", np.where(fold == 1, "val", "train")
    )
    ids_by_part = {
        part: sorted(identity.loc[identity["partition"] == part, "id"].tolist())
        for part in ("train", "val", "test")
    }
    split = {
        "schema_version": 2,
        "description": (
            "Frozen identity-grouped 8:1:1 split; fold 0 test, fold 1 "
            "validation, folds 2-9 training"
        ),
        "algorithm": "StratifiedGroupKFold",
        "seed": seed,
        "n_folds": N_FOLDS,
        "sizes": {part: len(values) for part, values in ids_by_part.items()},
        **ids_by_part,
    }
    audit = audit_confirmatory_split(ordered, identity, split)
    audit["identity_summary"] = identity_summary
    return split, identity, audit


def audit_confirmatory_split(
    df: pd.DataFrame,
    identity: pd.DataFrame,
    split: dict,
) -> dict:
    all_ids = set(df["id"].astype(str))
    parts = {part: set(split[part]) for part in ("train", "val", "test")}
    exhaustive = set.union(*parts.values()) == all_ids
    disjoint = all(
        parts[left].isdisjoint(parts[right])
        for index, left in enumerate(parts)
        for right in list(parts)[index + 1 :]
    )

    group_part_counts = identity.groupby("identity_group")["partition"].nunique()
    cross_partition_groups = group_part_counts[group_part_counts > 1]
    chemical = identity.dropna(subset=["chemical_identity_key"])
    chemical_part_counts = chemical.groupby("chemical_identity_key")[
        "partition"
    ].nunique()
    cross_partition_chemical = chemical_part_counts[chemical_part_counts > 1]

    checks = {
        "collectively_exhaustive": exhaustive,
        "mutually_disjoint": disjoint,
        "identity_groups_disjoint": cross_partition_groups.empty,
        "chemical_keys_disjoint": cross_partition_chemical.empty,
    }
    if not all(checks.values()):
        raise RuntimeError(f"confirmatory split audit failed: {checks}")
    return {
        "checks": checks,
        "sizes": {part: len(values) for part, values in parts.items()},
        "cross_partition_identity_groups": int(len(cross_partition_groups)),
        "cross_partition_chemical_keys": int(len(cross_partition_chemical)),
    }


def write_confirmatory_split(
    split: dict,
    identity: pd.DataFrame,
    audit: dict,
    out_dir: Path,
) -> None:
    """Write train/validation and locked-test IDs as separate artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    development = {
        key: split[key]
        for key in (
            "schema_version",
            "description",
            "algorithm",
            "seed",
            "n_folds",
        )
    }
    development["train"] = split["train"]
    development["val"] = split["val"]
    development["sizes"] = {
        "train": len(split["train"]),
        "val": len(split["val"]),
    }
    locked = {
        "schema_version": split["schema_version"],
        "description": "Locked confirmatory test IDs; training code must not load",
        "seed": split["seed"],
        "test": split["test"],
        "size": len(split["test"]),
    }
    (out_dir / "development.json").write_text(json.dumps(development, indent=2) + "\n")
    (out_dir / "locked_test_ids.json").write_text(json.dumps(locked, indent=2) + "\n")
    identity.to_csv(out_dir / "identity_groups.csv", index=False)
    (out_dir / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
