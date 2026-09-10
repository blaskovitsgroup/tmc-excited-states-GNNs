"""Phase 1: frozen data audit.

Joins tmQMg* (excited-state labels) to the tmQMg parent table and XYZ files,
validates integrity, builds eV/oscillator targets and masks, and writes the
frozen, auditable artifacts described in workflow section 16.
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from .. import config as C
from .targets import build_targets, detect_invalid_wavelengths
from .shift_audit import audit_source_shifts


def read_xyz_natoms(path: Path) -> int | None:
    """Atom count from an XYZ file's first line; None if unreadable."""
    try:
        with open(path) as fh:
            return int(fh.readline().strip())
    except (OSError, ValueError):
        return None


def load_star() -> pd.DataFrame:
    return pd.read_csv(C.STAR_CSV)


def load_parent() -> pd.DataFrame:
    return pd.read_csv(C.PARENT_CSV)


def index_xyz() -> dict[str, int]:
    """Map id -> atom count for every XYZ file present."""
    out: dict[str, int] = {}
    for p in sorted(C.XYZ_DIR.glob("*.xyz")):
        n = read_xyz_natoms(p)
        if n is not None:
            out[p.stem] = n
    return out


def build(
    star: pd.DataFrame, parent: pd.DataFrame, xyz_natoms: dict[str, int]
) -> dict[str, pd.DataFrame]:
    """Return cleaned table, id manifest, and invalid-label log."""
    parent_cols = ["id", "metal_center", "smiles"] + [
        c for c in C.PARENT_DESCRIPTOR_COLS if c in parent.columns
    ]
    merged = star.merge(parent[parent_cols], how="inner", on="id", validate="1:1")

    invalid = detect_invalid_wavelengths(merged)

    # XYZ availability + atom-count consistency vs parent n_atoms
    xyz_n = merged["id"].map(xyz_natoms)
    merged = pd.concat(
        [
            merged,
            pd.DataFrame(
                {"xyz_natoms": xyz_n, "has_xyz": xyz_n.notna()}, index=merged.index
            ),
        ],
        axis=1,
    )
    merged = build_targets(merged)
    natoms_mismatch = (
        merged["has_xyz"]
        & merged["n_atoms"].notna()
        & (merged["xyz_natoms"] != merged["n_atoms"])
    )
    for _id, nx, npar in zip(
        merged.loc[natoms_mismatch, "id"],
        merged.loc[natoms_mismatch, "xyz_natoms"],
        merged.loc[natoms_mismatch, "n_atoms"],
    ):
        invalid = pd.concat(
            [
                invalid,
                pd.DataFrame(
                    [
                        {
                            "id": _id,
                            "column": "n_atoms",
                            "value": f"xyz={nx} parent={npar}",
                            "reason": "xyz_parent_natoms_mismatch",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    manifest = pd.DataFrame(
        {
            "id": merged["id"],
            "metal": merged["metal_center"],
            "metal_row": merged["metal_center"].map(C.METAL_ROW),
            "charge": merged["charge"],
            "n_atoms": merged["n_atoms"],
            "n_electrons": merged["n_electrons"],
            "has_xyz": merged["has_xyz"],
            "has_parent": True,
            "has_visible_gas": merged["mask_peak_vis_gasphase"],
            "has_visible_acetone": merged["mask_peak_vis_acetone"],
            "ct_class_gas": merged["transition_nature_vis_gasphase"],
            "ct_class_acetone": merged["transition_nature_vis_acetone"],
            "smiles_present": merged["smiles"].notna(),
        }
    )

    shift_issues, shift_summary = audit_source_shifts(merged)
    return {
        "cleaned": merged,
        "manifest": manifest,
        "invalid": invalid,
        "shift_issues": shift_issues,
        "shift_summary": shift_summary,
    }


def run_audit(
    out_dir: Path | None = None, verbose: bool = True
) -> dict[str, pd.DataFrame]:
    """Run the full audit and write Phase 1 artifacts."""
    out_dir = Path(out_dir) if out_dir is not None else C.PHASE1_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[audit] loading star={C.STAR_CSV.name} parent={C.PARENT_CSV.name}")
    star = load_star()
    parent = load_parent()
    if verbose:
        print(f"[audit] indexing XYZ in {C.XYZ_DIR} ...")
    xyz_natoms = index_xyz()
    if verbose:
        print(f"[audit] star={len(star)} parent={len(parent)} xyz={len(xyz_natoms)}")

    arts = build(star, parent, xyz_natoms)

    cleaned_path = out_dir / "cleaned_tmQMg_star.parquet"
    arts["cleaned"].to_parquet(cleaned_path, index=False)
    arts["manifest"].to_csv(out_dir / "id_manifest.csv", index=False)
    arts["invalid"].to_csv(out_dir / "invalid_labels.csv", index=False)
    arts["shift_issues"].to_csv(
        out_dir / "excluded_source_shift_issues.csv", index=False
    )
    (out_dir / "excluded_source_shift_summary.json").write_text(
        json.dumps(arts["shift_summary"], indent=2) + "\n"
    )

    if verbose:
        n = len(arts["cleaned"])
        print(f"[audit] wrote cleaned rows={n} -> {cleaned_path}")
        print(f"[audit] invalid label cells flagged: {len(arts['invalid'])}")
        if len(arts["invalid"]):
            print(arts["invalid"].to_string(index=False))
        print(f"[audit] missing XYZ: {(~arts['manifest']['has_xyz']).sum()}")
        print(
            "[audit] source shift labels excluded; "
            f"issue rows={arts['shift_summary']['issue_rows']}"
        )
    return arts
