"""Evaluation metrics (numpy). Mask-aware; NaN entries are ignored.

Covers the state/oscillator/spectrum/solvatochromism/CT/uncertainty families of
workflow section 13. Inputs are plain numpy arrays so the same code scores both
classical baselines and neural models.
"""

from __future__ import annotations

import numpy as np

from ..config import HC_EV_NM


def _flat_valid(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None):
    p = np.asarray(pred, float).ravel()
    t = np.asarray(true, float).ravel()
    ok = np.isfinite(p) & np.isfinite(t)
    if mask is not None:
        ok &= np.asarray(mask, bool).ravel()
    return p[ok], t[ok]


def mae(pred, true, mask=None) -> float:
    p, t = _flat_valid(pred, true, mask)
    return float(np.mean(np.abs(p - t))) if p.size else float("nan")


def rmse(pred, true, mask=None) -> float:
    p, t = _flat_valid(pred, true, mask)
    return float(np.sqrt(np.mean((p - t) ** 2))) if p.size else float("nan")


def hit_rate(pred, true, tol: float, mask=None) -> float:
    p, t = _flat_valid(pred, true, mask)
    return float(np.mean(np.abs(p - t) <= tol)) if p.size else float("nan")


def energy_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """State-energy metrics in eV (pred/true shaped [N, n_states])."""
    out = {
        "mae_eV": mae(pred, true),
        "rmse_eV": rmse(pred, true),
        "hit_0.05eV": hit_rate(pred, true, 0.05),
        "hit_0.10eV": hit_rate(pred, true, 0.10),
        "hit_0.20eV": hit_rate(pred, true, 0.20),
    }
    # per-state MAE (state index 1..n)
    n = np.asarray(pred).shape[1]
    out["mae_per_state_eV"] = [
        mae(np.asarray(pred)[:, i], np.asarray(true)[:, i]) for i in range(n)
    ]
    # wavelength-space MAE (nm); only where both energies are physically positive
    p = np.asarray(pred, float)
    t = np.asarray(true, float)
    valid = np.isfinite(p) & np.isfinite(t) & (p > 1e-3) & (t > 1e-3)
    with np.errstate(divide="ignore", invalid="ignore"):
        pnm = np.where(valid, HC_EV_NM / p, np.nan)
        tnm = np.where(valid, HC_EV_NM / t, np.nan)
    out["mae_nm"] = mae(pnm, tnm)
    return out


def molecule_energy_errors(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """One mean absolute energy error per molecule.

    ``pred`` and ``true`` may be [N, S] or [N, solvent, S]. Missing state
    labels are ignored within each molecule.
    """
    prediction = np.asarray(pred, dtype=float)
    target = np.asarray(true, dtype=float)
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError(
            f"energy arrays must have equal shape [N, ...], got "
            f"{prediction.shape} and {target.shape}"
        )
    valid = np.isfinite(prediction) & np.isfinite(target)
    absolute = np.where(valid, np.abs(prediction - target), np.nan)
    axes = tuple(range(1, absolute.ndim))
    with np.errstate(invalid="ignore"):
        return np.nanmean(absolute, axis=axes)


def joint_molecule_energy_mae(pred: np.ndarray, true: np.ndarray) -> float:
    errors = molecule_energy_errors(pred, true)
    return float(np.nanmean(errors)) if np.isfinite(errors).any() else float("nan")


def oscillator_metrics(pred_logf: np.ndarray, true_logf: np.ndarray) -> dict:
    """Metrics on log1p(f); also Spearman on raw intensity ordering."""
    out = {
        "mae_logf": mae(pred_logf, true_logf),
        "rmse_logf": rmse(pred_logf, true_logf),
    }
    p, t = _flat_valid(pred_logf, true_logf)
    if p.size > 2:
        from scipy.stats import spearmanr

        out["spearman"] = float(spearmanr(p, t).statistic)
    return out


def gaussian_spectrum(
    energies: np.ndarray, f: np.ndarray, grid: np.ndarray, sigma: float = 0.2
) -> np.ndarray:
    """Broaden discrete (E, f) lines onto an energy grid (eV). Vectorized.

    energies/f shaped [N, S]; returns [N, len(grid)].
    """
    e = np.asarray(energies)[:, :, None]
    ff = np.clip(np.asarray(f), 0.0, 1e6)[:, :, None]
    g = grid[None, None, :]
    valid = np.isfinite(e) & np.isfinite(ff)
    e = np.where(valid, e, 0.0)
    ff = np.where(valid, ff, 0.0)
    return np.sum(ff * np.exp(-((g - e) ** 2) / (2 * sigma**2)), axis=1)


def spectrum_metrics(
    pred_E, pred_f, true_E, true_f, grid: np.ndarray | None = None, sigma: float = 0.2
) -> dict:
    if grid is None:
        grid = np.linspace(0.5, 8.0, 256)
    sp = gaussian_spectrum(pred_E, pred_f, grid, sigma)
    st = gaussian_spectrum(true_E, true_f, grid, sigma)
    cos = np.sum(sp * st, axis=1) / (
        np.linalg.norm(sp, axis=1) * np.linalg.norm(st, axis=1) + 1e-12
    )
    iae = np.sum(np.abs(sp - st), axis=1) / (np.sum(st, axis=1) + 1e-12)
    return {
        "spectrum_cosine": float(np.nanmean(cos)),
        "spectrum_iae": float(np.nanmean(iae)),
    }


def classification_metrics(
    pred_labels: np.ndarray, true_labels: np.ndarray, classes: list[str]
) -> dict:
    """Accuracy, balanced accuracy, macro-F1, and confusion matrix."""
    p = np.asarray(pred_labels)
    t = np.asarray(true_labels)
    ok = t != None  # noqa: E711 - object array compare
    p, t = p[ok], t[ok]
    if p.size == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "confusion_matrix": [],
        }
    acc = float(np.mean(p == t))
    f1s = []
    recalls = []
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    class_index = {label: index for index, label in enumerate(classes)}
    for predicted, target in zip(p, t):
        if predicted in class_index and target in class_index:
            confusion[class_index[target], class_index[predicted]] += 1
    for c in classes:
        tp = np.sum((p == c) & (t == c))
        fp = np.sum((p == c) & (t != c))
        fn = np.sum((p != c) & (t == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        recalls.append(rec)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {
        "accuracy": acc,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": confusion.tolist(),
    }


def expected_calibration_error(
    probability: np.ndarray, target: np.ndarray, n_bins: int = 10
) -> float:
    p, t = _flat_valid(probability, target)
    if p.size == 0:
        return float("nan")
    p = np.clip(p, 0, 1)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    value = 0.0
    for bin_index in range(n_bins):
        selected = bins == bin_index
        if selected.any():
            value += selected.mean() * abs(p[selected].mean() - t[selected].mean())
    return float(value)


def binary_classification_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    p, t = _flat_valid(probability, target)
    if p.size == 0 or len(np.unique(t)) < 2:
        auroc = float("nan")
        average_precision = float("nan")
    else:
        from sklearn.metrics import average_precision_score, roc_auc_score

        auroc = float(roc_auc_score(t, p))
        average_precision = float(average_precision_score(t, p))
    predicted = p >= threshold
    truth = t.astype(bool)
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "auroc": auroc,
        "average_precision": average_precision,
        "brier": float(np.mean((p - t) ** 2)) if p.size else float("nan"),
        "ece_10bin": expected_calibration_error(p, t, n_bins=10),
        "accuracy": float(np.mean(predicted == truth)) if p.size else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "threshold": threshold,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def solvent_shift_metrics(
    gas_prediction: np.ndarray,
    acetone_prediction: np.ndarray,
    gas_target: np.ndarray,
    acetone_target: np.ndarray,
) -> dict:
    """Statewise acetone-minus-gas energy-shift metrics in eV."""
    predicted_shift = np.asarray(acetone_prediction) - np.asarray(gas_prediction)
    target_shift = np.asarray(acetone_target) - np.asarray(gas_target)
    return {
        "mae_eV": mae(predicted_shift, target_shift),
        "rmse_eV": rmse(predicted_shift, target_shift),
        "mae_per_state_eV": [
            mae(predicted_shift[:, index], target_shift[:, index])
            for index in range(predicted_shift.shape[1])
        ],
    }


def nll_gaussian(pred_mean, pred_var, true) -> float:
    """Mean Gaussian negative log likelihood for calibrated uncertainty.

    All three arrays are masked JOINTLY on finiteness — slicing the variance to
    the filtered length instead would silently misalign it after any NaN.
    """
    m = np.asarray(pred_mean, float).ravel()
    t = np.asarray(true, float).ravel()
    v = np.asarray(pred_var, float).ravel() + 1e-6
    ok = np.isfinite(m) & np.isfinite(t) & np.isfinite(v)
    m, t, v = m[ok], t[ok], v[ok]
    return float(np.mean(0.5 * (np.log(2 * np.pi * v) + (t - m) ** 2 / v)))


def coverage(pred_mean, pred_std, true, z: float = 1.0) -> float:
    """Fraction of truths within +-z std of the prediction (jointly masked)."""
    m = np.asarray(pred_mean, float).ravel()
    t = np.asarray(true, float).ravel()
    s = np.asarray(pred_std, float).ravel()
    ok = np.isfinite(m) & np.isfinite(t) & np.isfinite(s)
    m, t, s = m[ok], t[ok], s[ok]
    return float(np.mean(np.abs(t - m) <= z * s))
