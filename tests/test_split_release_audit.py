import json

import pandas as pd

from tmqmg_es.data.split_release_audit import build_audit


def test_split_release_audit_is_json_serializable(tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    pd.DataFrame(
        {
            "id": ["A", "B", "C"],
            "metal_center": ["Fe", "Ru", "Ir"],
            "charge": [0, 1, -1],
            "n_atoms": [20, 30, 40],
            "mask_peak_vis_gasphase": [True, False, True],
            "mask_peak_vis_acetone": [True, True, False],
        }
    ).to_parquet(tmp_path / "cleaned_tmQMg_star.parquet", index=False)
    pd.DataFrame(
        {
            "id": ["A", "B", "C"],
            "identity_group": ["A", "B", "C"],
            "partition": ["train", "val", "test"],
            "fold": [2, 1, 0],
        }
    ).to_csv(split_dir / "identity_groups.csv", index=False)
    (split_dir / "split_audit.json").write_text(
        json.dumps({"identity_summary": {"near_geometry_pairs": []}})
    )
    audit = build_audit(tmp_path)
    json.dumps(audit)
    assert all(audit["checks"].values())
