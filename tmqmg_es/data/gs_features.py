"""The nine frozen ground-state descriptors used by XGBoost."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C


def _assert_no_leakage(cols: list[str]) -> None:
    leaked = set(cols) & set(C.LEAKAGE_COLS)
    if leaked:
        raise ValueError(f"leakage columns in feature set: {sorted(leaked)}")


def build_feature_frame(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by id with numeric ground-state features."""
    base_cols = list(C.PARENT_DESCRIPTOR_COLS)
    missing = set(base_cols) - set(cleaned.columns)
    if missing:
        raise ValueError(f"missing frozen descriptor columns: {sorted(missing)}")
    _assert_no_leakage(base_cols)
    return cleaned[["id"] + base_cols].set_index("id").copy()


def feature_matrix(feats: pd.DataFrame, ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Numeric matrix for the given ids; missing values median-imputed later."""
    sub = feats.loc[ids]
    names = list(sub.columns)
    _assert_no_leakage(names)
    return sub.to_numpy(dtype=float), names


def target_matrix(
    cleaned: pd.DataFrame, ids: list[str], solvent: str = "gasphase"
) -> np.ndarray:
    """30 excitation energies (eV) for the given ids and solvent."""
    cols = [f"E_{i}_{solvent}_eV" for i in range(1, C.N_STATES + 1)]
    return cleaned.set_index("id").loc[ids, cols].to_numpy(dtype=float)


class Standardizer:
    """Column standardizer with NaN-safe median imputation (fit on train)."""

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.median_ = np.nanmedian(x, axis=0)
        xi = np.where(np.isnan(x), self.median_, x)
        self.mean_ = xi.mean(axis=0)
        self.std_ = xi.std(axis=0) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        xi = np.where(np.isnan(x), self.median_, x)
        return (xi - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)
