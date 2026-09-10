from pathlib import Path

import numpy as np
import pandas as pd

from tmqmg_es.data import splits as S


def _toy_df(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    smiles = [f"[Fe](N)(N)(N)(N)C{i}" for i in range(n)]
    return pd.DataFrame(
        {
            "id": [f"M{i:04d}" for i in range(n)],
            "metal_center": rng.choice(["Fe", "Ru", "Ir", "Pd"], n),
            "charge": rng.choice([-1, 0, 1], n),
            "n_atoms": rng.integers(10, 80, n),
            "smiles": smiles,
            "mask_peak_vis_gasphase": rng.random(n) > 0.5,
            "mask_peak_vis_acetone": rng.random(n) > 0.5,
        }
    )


def _write_xyz(path: Path, atoms: list[tuple[str, float, float, float]]) -> None:
    lines = [str(len(atoms)), path.stem]
    lines.extend(f"{symbol} {x:.8f} {y:.8f} {z:.8f}" for symbol, x, y, z in atoms)
    path.write_text("\n".join(lines) + "\n")


def test_canonical_identity_respects_charge():
    key_a = S.canonical_identity_key("C(C)O", 0)
    key_b = S.canonical_identity_key("OCC", 0)
    key_c = S.canonical_identity_key("OCC", 1)
    assert key_a == key_b
    assert key_a != key_c
    assert S.canonical_identity_key(None, 0) is None


def test_kabsch_rmsd_handles_rotation_translation_and_atom_reordering():
    z_ref = np.array([26, 7, 7, 6])
    pos_ref = np.array(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.7, 0.0], [0.0, 0.0, 1.3]]
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    permutation = np.array([2, 0, 3, 1])
    z_query = z_ref[permutation]
    pos_query = (pos_ref @ rotation + np.array([4.0, -2.0, 1.0]))[permutation]
    assert S.kabsch_rmsd(z_ref, pos_ref, z_query, pos_query) < 1e-7


def test_identity_groups_union_equal_chemistry_and_geometry(tmp_path):
    frame = pd.DataFrame(
        {
            "id": ["A", "B", "C", "D"],
            "smiles": ["CCO", "OCC", None, None],
            "charge": [0, 0, 1, 1],
        }
    )
    atoms = [
        ("Fe", 0.0, 0.0, 0.0),
        ("N", 1.5, 0.0, 0.0),
        ("N", 0.0, 1.5, 0.0),
    ]
    _write_xyz(tmp_path / "A.xyz", atoms)
    _write_xyz(tmp_path / "B.xyz", atoms)
    _write_xyz(tmp_path / "C.xyz", atoms)
    shifted = [(symbol, x + 5.0, y - 2.0, z + 1.0) for symbol, x, y, z in atoms]
    _write_xyz(tmp_path / "D.xyz", shifted)
    groups, summary = S.build_identity_groups(frame, xyz_dir=tmp_path)
    assert groups["identity_group"].nunique() == 1
    assert summary["n_non_singleton_groups"] == 1


def test_confirmatory_split_is_clean_and_deterministic(tmp_path):
    frame = _toy_df()
    labels = S._stratification_labels(frame)
    assert labels.value_counts().min() >= S.N_FOLDS
    split_a, identity_a, audit_a = S.build_confirmatory_split(
        frame, xyz_dir=tmp_path, seed=S.SPLIT_SEED
    )
    split_b, identity_b, audit_b = S.build_confirmatory_split(
        frame, xyz_dir=tmp_path, seed=S.SPLIT_SEED
    )
    assert split_a == split_b
    pd.testing.assert_frame_equal(identity_a, identity_b)
    assert audit_a == audit_b

    parts = {name: set(split_a[name]) for name in ("train", "val", "test")}
    assert set.union(*parts.values()) == set(frame["id"])
    assert parts["train"].isdisjoint(parts["val"])
    assert parts["train"].isdisjoint(parts["test"])
    assert parts["val"].isdisjoint(parts["test"])
    assert identity_a.groupby("identity_group")["partition"].nunique().max() == 1


def test_confirmatory_seed_cannot_be_changed(tmp_path):
    with np.testing.assert_raises(ValueError):
        S.build_confirmatory_split(_toy_df(), xyz_dir=tmp_path, seed=0)
