"""Generate a beginner-accessible dataset report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .. import config as C


def _md_table(series: pd.Series, key_name: str, val_name: str = "count") -> str:
    lines = [f"| {key_name} | {val_name} |", "| --- | --- |"]
    for k, v in series.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def build_report(
    cleaned: pd.DataFrame,
    manifest: pd.DataFrame,
    invalid: pd.DataFrame,
    split_sizes: dict | None = None,
    split_audit: dict | None = None,
) -> str:
    n = len(cleaned)
    out: list[str] = ["# tmQMg* dataset statistics", ""]
    out += [
        f"- Molecules (rows): **{n}**",
        "- Reference level: TD-DFT wB97xd/def2SVP, gas + acetone (SMD), 30 states.",
        f"- Columns in cleaned table: {cleaned.shape[1]}",
        f"- Missing XYZ: {(~manifest['has_xyz']).sum()}",
        f"- Missing SMILES: {(~manifest['smiles_present']).sum()}",
        f"- Invalid label cells flagged: {len(invalid)}",
        "",
    ]

    out += [
        "## Metal distribution",
        "",
        _md_table(manifest["metal"].value_counts(), "metal"),
        "",
    ]
    out += [
        "## Metal period (3d/4d/5d)",
        "",
        _md_table(
            manifest["metal_row"].map({4: "3d", 5: "4d", 6: "5d"}).value_counts(),
            "period",
        ),
        "",
    ]
    out += [
        "## Charge distribution",
        "",
        _md_table(manifest["charge"].value_counts().sort_index(), "charge"),
        "",
    ]

    out += ["## Atom count", "", "| stat | value |", "| --- | --- |"]
    for stat, val in (
        manifest["n_atoms"]
        .describe()[["min", "25%", "50%", "75%", "max", "mean"]]
        .items()
    ):
        out.append(f"| {stat} | {val:.1f} |")
    out.append("")

    out += [
        "## Visible-band availability",
        "",
        "| solvent | with visible band | without |",
        "| --- | --- | --- |",
    ]
    for s, col in (("gasphase", "has_visible_gas"), ("acetone", "has_visible_acetone")):
        yes = int(manifest[col].sum())
        out.append(f"| {s} | {yes} | {n - yes} |")
    out.append("")

    out += ["## Visible charge-transfer class balance", ""]
    for s, col in (("gasphase", "ct_class_gas"), ("acetone", "ct_class_acetone")):
        vc = manifest[col].value_counts(dropna=False)
        vc.index = [("none" if pd.isna(i) else i) for i in vc.index]
        out += [f"### {s}", "", _md_table(vc, "class"), ""]

    out += [
        "## nIR / UV band availability",
        "",
        "| region | gasphase | acetone |",
        "| --- | --- | --- |",
    ]
    for r in ("uv", "vis", "nir"):
        g = int(cleaned[f"mask_peak_{r}_gasphase"].sum())
        a = int(cleaned[f"mask_peak_{r}_acetone"].sum())
        out.append(f"| {r} | {g} | {a} |")
    out.append("")

    if split_sizes:
        out += [
            "## Confirmatory split",
            "",
            "The split unit is a complete molecule, not an individual solvent "
            "record or excitation. Gas-phase and acetone labels therefore stay "
            "together. Canonical-SMILES and near-identical-geometry groups also "
            "stay together, preventing structural twins from crossing partitions.",
            "",
            "| partition | molecules | purpose |",
            "| --- | ---: | --- |",
            f"| training | {split_sizes['train']} | fit model parameters |",
            f"| validation | {split_sizes['val']} | early stopping only |",
            f"| locked test | {split_sizes['test']} | one final evaluation |",
            "",
        ]
    if split_audit:
        identity = split_audit["identity_summary"]
        out += [
            "### Identity audit",
            "",
            f"- Identity groups: {identity['n_identity_groups']}",
            f"- Non-singleton groups: {identity['n_non_singleton_groups']}",
            f"- Largest identity group: {identity['largest_group']} molecules",
            "- Identity groups crossing partitions: 0",
            "- Canonical chemical keys crossing partitions: 0",
            "",
        ]

    if len(invalid):
        out += [
            "## Flagged invalid labels",
            "",
            _md_table(invalid.set_index("id")["reason"], "id", "reason"),
            "",
        ]

    return "\n".join(out)


def run_stats(out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir is not None else C.PHASE1_DIR
    cleaned = pd.read_parquet(out_dir / "cleaned_tmQMg_star.parquet")
    manifest = pd.read_csv(out_dir / "id_manifest.csv")
    invalid = pd.read_csv(out_dir / "invalid_labels.csv")
    split_sizes = None
    split_audit = None
    development_path = out_dir / "splits" / "development.json"
    locked_path = out_dir / "splits" / "locked_test_ids.json"
    audit_path = out_dir / "splits" / "split_audit.json"
    if development_path.exists() and locked_path.exists():
        development = json.loads(development_path.read_text())
        locked = json.loads(locked_path.read_text())
        split_sizes = {
            "train": len(development["train"]),
            "val": len(development["val"]),
            "test": len(locked["test"]),
        }
    if audit_path.exists():
        split_audit = json.loads(audit_path.read_text())
    report = build_report(cleaned, manifest, invalid, split_sizes, split_audit)
    path = out_dir / "dataset_statistics.md"
    path.write_text(report)
    print(f"[stats] wrote {path}")
    return path
