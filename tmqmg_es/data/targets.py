"""Label preprocessing: eV conversion, oscillator transforms, masks, NTO.

All functions take and return a pandas DataFrame and add derived columns. Raw
wavelength / oscillator columns are kept untouched for reporting (workflow
section 7). Derived columns are assembled in one concat to avoid DataFrame
fragmentation.

Mask convention: a region "band" exists exactly when the dataset reports a
``lambda_max_<region>_<solvent>``. The original parser
(tddft_data_parser.py::_get_lambda_max) returns None for lambda/f/sigma
together whenever f_max < 0.01, and the NTO/CT labels are gated on the same
condition. We therefore key masks on ``lambda_max.notna()`` rather than
recomputing the f_max > F_THRESHOLD test (which disagrees on the boundary
f_max == 0.01).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CT_CLASSES, HC_EV_NM, N_STATES, REGIONS, SOLVENTS


def nm_to_ev(lambda_nm: pd.Series | np.ndarray) -> np.ndarray:
    """E[eV] = HC / lambda[nm]; non-physical wavelengths (<=0) -> NaN."""
    lam = np.asarray(lambda_nm, dtype=float)
    out = np.full_like(lam, np.nan)
    valid = np.isfinite(lam) & (lam > 0)
    out[valid] = HC_EV_NM / lam[valid]
    return out


def detect_invalid_wavelengths(df: pd.DataFrame) -> pd.DataFrame:
    """Flag non-physical state wavelengths (lambda <= 0). Returns long table."""
    records = []
    for s in SOLVENTS:
        for i in range(1, N_STATES + 1):
            col = f"lambda_{i}_{s}"
            if col not in df.columns:
                continue
            bad = df[col] <= 0
            for _id, val in zip(df.loc[bad, "id"], df.loc[bad, col]):
                records.append(
                    {
                        "id": _id,
                        "column": col,
                        "value": val,
                        "reason": "non_physical_wavelength_le_0",
                    }
                )
    return pd.DataFrame(records, columns=["id", "column", "value", "reason"])


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add eV energies, log1p oscillators, masks, and NTO fractions.

    Returns a new de-fragmented DataFrame.
    """
    derived: dict[str, np.ndarray] = {}

    for s in SOLVENTS:
        # state-resolved energies (eV) and log1p oscillator strengths
        for i in range(1, N_STATES + 1):
            derived[f"E_{i}_{s}_eV"] = nm_to_ev(df[f"lambda_{i}_{s}"])
            derived[f"logf_{i}_{s}"] = np.log1p(
                df[f"f_{i}_{s}"].clip(lower=0).to_numpy()
            )

        # region peak energies + masks (band exists <=> lambda_max present)
        for r in REGIONS:
            lcol = f"lambda_max_{r}_{s}"
            derived[f"E_max_{r}_{s}_eV"] = nm_to_ev(df[lcol])
            derived[f"mask_peak_{r}_{s}"] = df[lcol].notna().to_numpy()

        # visible CT class + NTO contributions are gated on the visible band
        visible = derived[f"mask_peak_vis_{s}"]
        derived[f"mask_ct_{s}"] = (
            visible & df[f"transition_nature_vis_{s}"].isin(CT_CLASSES).to_numpy()
        )
        nto_columns = [
            f"{prefix}_contribution_{kind}_{s}"
            for kind in ("occupied", "virtual")
            for prefix in ("M", "L")
        ]
        nto_finite = np.isfinite(df[nto_columns].to_numpy(dtype=float)).all(axis=1)
        derived[f"mask_nto_{s}"] = visible & nto_finite
        for kind, short in (("occupied", "occ"), ("virtual", "virt")):
            metal = df[f"M_contribution_{kind}_{s}"].to_numpy(dtype=float)
            ligand = df[f"L_contribution_{kind}_{s}"].to_numpy(dtype=float)
            denominator = metal + ligand
            derived[f"M_frac_{short}_{s}"] = np.where(
                denominator > 0, metal / denominator, np.nan
            )

    return pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)
