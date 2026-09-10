import json

import numpy as np

from tmqmg_es.provenance import sha256_file
from tmqmg_es.train.ensemble_manifest import MODEL_IDS
from tmqmg_es.train.locked_eval import summarize_locked_predictions
from tmqmg_es.train.trainer import FROZEN_SEEDS


def _synthetic_arrays(n_members: int, n_molecules: int, neural: bool) -> dict:
    energy_target = np.full((n_molecules, 2, 30), 2.0, dtype=np.float32)
    offsets = np.linspace(-0.1, 0.1, n_members, dtype=np.float32)
    energy_prediction = np.stack(
        [energy_target + offset for offset in offsets],
        axis=0,
    )
    arrays = {
        "energy_prediction": energy_prediction,
        "energy_target": energy_target,
    }
    if not neural:
        return arrays

    peak_energy = np.empty((n_members, n_molecules, 2, 3), dtype=np.float32)
    peak_energy[..., 0] = 4.0
    peak_energy[..., 1] = 2.0
    peak_energy[..., 2] = 1.0
    arrays.update(
        {
            "logf_prediction": np.zeros(
                (n_members, n_molecules, 2, 30), dtype=np.float32
            ),
            "band_probability": np.full(
                (n_members, n_molecules, 2, 3), 0.5, dtype=np.float32
            ),
            "peak_energy_prediction": peak_energy,
            "peak_logf_prediction": np.zeros(
                (n_members, n_molecules, 2, 3), dtype=np.float32
            ),
            "peak_logwidth_prediction": np.zeros(
                (n_members, n_molecules, 2, 3), dtype=np.float32
            ),
            "ct_probability": np.full(
                (n_members, n_molecules, 2, 4), 0.25, dtype=np.float32
            ),
            "nto_prediction": np.full(
                (n_members, n_molecules, 2, 2), 0.5, dtype=np.float32
            ),
            "logf_target": np.zeros((n_molecules, 2, 30), dtype=np.float32),
            "band_target": np.zeros((n_molecules, 2, 3), dtype=np.float32),
            "peak_energy_target": np.full(
                (n_molecules, 2, 3), np.nan, dtype=np.float32
            ),
            "peak_logf_target": np.full((n_molecules, 2, 3), np.nan, dtype=np.float32),
            "peak_logwidth_target": np.full(
                (n_molecules, 2, 3), np.nan, dtype=np.float32
            ),
            "peak_mask": np.zeros((n_molecules, 2, 3), dtype=bool),
            # Production CT targets share a floating-point target tensor even
            # though the masked values encode integer class indices.
            "ct_target": np.full((n_molecules, 2), -1, dtype=np.float32),
            "ct_mask": np.zeros((n_molecules, 2), dtype=bool),
            "nto_target": np.full((n_molecules, 2, 2), np.nan, dtype=np.float32),
            "nto_mask": np.zeros((n_molecules, 2), dtype=bool),
        }
    )
    arrays["ct_target"][0, :] = 1.0
    arrays["ct_mask"][0, :] = True
    return arrays


