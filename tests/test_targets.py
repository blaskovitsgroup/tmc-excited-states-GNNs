import numpy as np
import pandas as pd

from tmqmg_es.config import HC_EV_NM
from tmqmg_es.data import targets as T


def test_nm_to_ev_roundtrip():
    lam = np.array([200.0, 400.0, 825.0])
    ev = T.nm_to_ev(lam)
    assert np.allclose(ev, HC_EV_NM / lam)
    # eV back to nm
    assert np.allclose(HC_EV_NM / ev, lam)


def test_nm_to_ev_invalid_to_nan():
    out = T.nm_to_ev(np.array([-4321.96, 0.0, np.nan, 500.0]))
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[2])
    assert np.isclose(out[3], HC_EV_NM / 500.0)


def test_detect_invalid_wavelengths():
    df = pd.DataFrame(
        {
            "id": ["A", "B"],
            "lambda_1_gasphase": [500.0, 600.0],
            "lambda_1_acetone": [-4321.96, 700.0],
        }
    )
    bad = T.detect_invalid_wavelengths(df)
    assert list(bad["id"]) == ["A"]
    assert bad.iloc[0]["column"] == "lambda_1_acetone"


def _toy_full_row():
    """A minimal row with all columns build_targets touches."""
    from tmqmg_es.config import N_STATES, REGIONS, SOLVENTS

    row = {"id": "X"}
    for s in SOLVENTS:
        for i in range(1, N_STATES + 1):
            row[f"lambda_{i}_{s}"] = 400.0 + i
            row[f"f_{i}_{s}"] = 0.1
        for r in REGIONS:
            row[f"lambda_max_{r}_{s}"] = 500.0 if r == "vis" else np.nan
            row[f"f_max_{r}_{s}"] = 0.5 if r == "vis" else np.nan
            row[f"sigma_{r}_{s}"] = 1.0 if r == "vis" else np.nan
        row[f"transition_nature_vis_{s}"] = "MLCT"
        for kind in ("occupied", "virtual"):
            row[f"M_contribution_{kind}_{s}"] = 0.7
            row[f"L_contribution_{kind}_{s}"] = 0.3
    return pd.DataFrame([row])


def test_build_targets_masks_and_fracs():
    df = T.build_targets(_toy_full_row())
    assert np.isclose(df["E_1_gasphase_eV"].iloc[0], HC_EV_NM / 401.0)
    assert np.isclose(df["logf_1_gasphase"].iloc[0], np.log1p(0.1))
    # vis band present, uv/nir absent
    assert bool(df["mask_peak_vis_gasphase"].iloc[0]) is True
    assert bool(df["mask_peak_uv_gasphase"].iloc[0]) is False
    # NTO collapsed to metal fraction 0.7/(0.7+0.3)
    assert np.isclose(df["M_frac_occ_gasphase"].iloc[0], 0.7)
    assert bool(df["mask_ct_gasphase"].iloc[0]) is True
