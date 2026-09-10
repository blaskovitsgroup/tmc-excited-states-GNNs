"""Independently recompute energy MAEs from published prediction arrays."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    from retrain import MODELS
    result = {}
    for model in MODELS:
        with np.load(ROOT / "outputs/locked_predictions" / f"{model}.npz") as data:
            predicted = data["energy_prediction"].mean(axis=0)
            target = data["energy_target"]
            errors = np.abs(predicted - target)
            errors[~np.isfinite(target)] = np.nan
            result[model] = {
                "gasphase_mae_eV": float(np.nanmean(np.nanmean(errors[:, 0], axis=1))),
                "acetone_mae_eV": float(np.nanmean(np.nanmean(errors[:, 1], axis=1))),
                "joint_molecule_mae_eV": float(np.nanmean(np.nanmean(errors, axis=(1, 2)))),
            }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
