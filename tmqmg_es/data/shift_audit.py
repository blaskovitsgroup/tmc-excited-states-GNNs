"""Recompute source shift labels for auditing; does not train on them."""

from __future__ import annotations


import numpy as np
import pandas as pd

from .. import config as C

RELATIONS = ("vis_to_vis", "uv_to_vis", "nir_to_vis", "vis_to_uv", "vis_to_nir")


def _available(row: pd.Series, region: str, solvent: str) -> bool:
    oscillator = row[f"f_max_{region}_{solvent}"]
    wavelength = row[f"lambda_max_{region}_{solvent}"]
    return (
        np.isfinite(oscillator)
        and oscillator >= C.F_THRESHOLD
        and np.isfinite(wavelength)
        and wavelength > 0
    )


def _nearest_region(
    row: pd.Series,
    source_solvent: str,
    target_wavelength: float,
) -> str | None:
    candidates = []
    for region in ("uv", "nir"):
        if _available(row, region, source_solvent):
            wavelength = row[f"lambda_max_{region}_{source_solvent}"]
            candidates.append((abs(wavelength - target_wavelength), region))
    return min(candidates)[1] if candidates else None


def corrected_visible_shift(row: pd.Series) -> dict:
    """Return a physically defined matched-band audit record.

    The value is undefined when neither phase has a visible band or when a
    visible/non-visible transition cannot be matched to a reported UV/NIR band.
    Undefined is represented by NaN, never by zero.
    """
    gas_visible = _available(row, "vis", "gasphase")
    acetone_visible = _available(row, "vis", "acetone")
    relation = None
    source_region = None
    target_region = None

    if gas_visible and acetone_visible:
        relation = "vis_to_vis"
        source_region = target_region = "vis"
    elif not gas_visible and acetone_visible:
        source_region = _nearest_region(row, "gasphase", row["lambda_max_vis_acetone"])
        target_region = "vis"
        if source_region is not None:
            relation = f"{source_region}_to_vis"
    elif gas_visible and not acetone_visible:
        source_region = "vis"
        target_region = _nearest_region(row, "acetone", row["lambda_max_vis_gasphase"])
        if target_region is not None:
            relation = f"vis_to_{target_region}"

    if relation is None:
        return {
            "corrected_lambda_delta": np.nan,
            "corrected_f_delta": np.nan,
            "corrected_relation": None,
            "corrected_shift_defined": False,
        }
    source_lambda = row[f"lambda_max_{source_region}_gasphase"]
    target_lambda = row[f"lambda_max_{target_region}_acetone"]
    source_f = row[f"f_max_{source_region}_gasphase"]
    target_f = row[f"f_max_{target_region}_acetone"]
    return {
        "corrected_lambda_delta": target_lambda - source_lambda,
        "corrected_f_delta": target_f - source_f,
        "corrected_relation": relation,
        "corrected_shift_defined": True,
    }


def audit_source_shifts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    corrected = pd.DataFrame(
        [corrected_visible_shift(row) for _, row in df.iterrows()],
        index=df.index,
    )
    source_relation = pd.Series(None, index=df.index, dtype=object)
    for relation in RELATIONS:
        active = df[relation].fillna(False).astype(bool)
        source_relation.loc[active] = relation
    audit = pd.concat(
        [
            df[["id", "lambda_delta", "f_delta"] + list(RELATIONS)].copy(),
            source_relation.rename("source_relation"),
            corrected,
        ],
        axis=1,
    )
    numeric_mismatch = corrected["corrected_shift_defined"] & (
        ~np.isclose(
            df["lambda_delta"],
            corrected["corrected_lambda_delta"],
            equal_nan=True,
        )
        | ~np.isclose(df["f_delta"], corrected["corrected_f_delta"], equal_nan=True)
    )
    relation_mismatch = source_relation.fillna("__undefined__") != corrected[
        "corrected_relation"
    ].fillna("__undefined__")
    undefined_stored_as_zero = (
        ~corrected["corrected_shift_defined"]
        & np.isclose(df["lambda_delta"], 0, equal_nan=False)
        & np.isclose(df["f_delta"], 0, equal_nan=False)
    )
    audit["numeric_mismatch"] = numeric_mismatch
    audit["relation_mismatch"] = relation_mismatch
    audit["undefined_stored_as_zero"] = undefined_stored_as_zero
    issues = audit[
        numeric_mismatch | relation_mismatch | undefined_stored_as_zero
    ].copy()
    summary = {
        "rows": len(df),
        "corrected_shift_defined": int(corrected["corrected_shift_defined"].sum()),
        "numeric_mismatches": int(numeric_mismatch.sum()),
        "relation_mismatches": int(relation_mismatch.sum()),
        "undefined_stored_as_zero": int(undefined_stored_as_zero.sum()),
        "issue_rows": len(issues),
        "training_use": "excluded",
        "reason": (
            "derives solvent energy shifts from paired state predictions and "
            "does not optimize source lambda_delta/f_delta labels"
        ),
    }
    return issues, summary
