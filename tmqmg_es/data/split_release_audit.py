"""Structural and covariate-balance audit for a frozen split."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _marginal_audit(frame: pd.DataFrame, column: str) -> dict:
    overall = frame[column].value_counts(normalize=True, dropna=False)
    by_partition = pd.crosstab(
        frame[column], frame["partition"], normalize="columns", dropna=False
    )
    deviation = by_partition.sub(overall, axis=0).abs()
    return {
        "max_absolute_proportion_deviation": float(deviation.max().max()),
        "overall_proportions": {
            str(key): float(value) for key, value in overall.items()
        },
        "partition_proportions": {
            str(partition): {
                str(key): float(value) for key, value in by_partition[partition].items()
            }
            for partition in by_partition
        },
    }


def build_audit(phase1: Path) -> dict:
    cleaned = pd.read_parquet(phase1 / "cleaned_tmQMg_star.parquet")
    identity = pd.read_csv(phase1 / "splits" / "identity_groups.csv")
    frame = cleaned.merge(
        identity[["id", "identity_group", "partition", "fold"]],
        on="id",
        validate="one_to_one",
    )
    split_audit = json.loads((phase1 / "splits" / "split_audit.json").read_text())
    near_pairs = split_audit["identity_summary"]["near_geometry_pairs"]
    partition_by_id = frame.set_index("id")["partition"]
    checks = {
        "row_count_matches": bool(len(frame) == len(cleaned) == len(identity)),
        "unique_molecule_ids": bool(not frame["id"].duplicated().any()),
        "identity_groups_disjoint": bool(
            frame.groupby("identity_group")["partition"].nunique().max() == 1
        ),
        "near_geometry_pairs_disjoint": bool(
            all(
                partition_by_id[pair["left"]] == partition_by_id[pair["right"]]
                for pair in near_pairs
            )
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen split release audit failed: {checks}")

    partition_summary = {}
    for partition, subset in frame.groupby("partition", sort=True):
        partition_summary[partition] = {
            "molecules": int(len(subset)),
            "atom_count_mean": float(subset["n_atoms"].mean()),
            "atom_count_median": float(subset["n_atoms"].median()),
            "gas_visible_fraction": float(subset["mask_peak_vis_gasphase"].mean()),
            "acetone_visible_fraction": float(subset["mask_peak_vis_acetone"].mean()),
        }
    return {
        "schema_version": 1,
        "checks": checks,
        "n_molecules": int(len(frame)),
        "n_identity_groups": int(frame["identity_group"].nunique()),
        "n_verified_near_geometry_pairs": int(len(near_pairs)),
        "maximum_verified_rmsd_angstrom": float(
            max((pair["rmsd_angstrom"] for pair in near_pairs), default=0.0)
        ),
        "partition_summary": partition_summary,
        "marginal_balance": {
            column: _marginal_audit(frame, column)
            for column in (
                "metal_center",
                "charge",
                "mask_peak_vis_gasphase",
                "mask_peak_vis_acetone",
            )
        },
    }
