"""Validation-only audit of every output head in selected neural checkpoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..models.heads import VIS_HI_EV, VIS_LO_EV
from ..provenance import object_sha256, sha256_file
from .ensemble_manifest import _load_valid_member
from .locked_eval import _predict_neural, _validate_prediction_arrays
from .trainer import FROZEN_SEEDS, MODEL_SPECS


def _assert_neural_constraints(arrays: dict[str, np.ndarray], model_id: str) -> None:
    energy = arrays["energy_prediction"]
    if np.any(energy <= 0.0) or np.any(energy > 15.0):
        raise RuntimeError(f"{model_id} selected validation energy outside (0, 15]")
    if np.any(arrays["logf_prediction"] < 0.0):
        raise RuntimeError(f"{model_id} has negative log1p oscillator predictions")
    if np.any(arrays["peak_logf_prediction"] < 0.0):
        raise RuntimeError(f"{model_id} has negative peak log1p intensity predictions")
    if np.any(arrays["peak_logwidth_prediction"] < 0.0):
        raise RuntimeError(f"{model_id} has negative peak log-width predictions")

    peak_energy = arrays["peak_energy_prediction"]
    uv = peak_energy[..., 0]
    visible = peak_energy[..., 1]
    nir = peak_energy[..., 2]
    if np.any(uv <= VIS_HI_EV):
        raise RuntimeError(f"{model_id} has UV peak predictions outside the UV region")
    if np.any(visible < VIS_LO_EV) or np.any(visible > VIS_HI_EV):
        raise RuntimeError(
            f"{model_id} has visible peak predictions outside the visible region"
        )
    if np.any(nir <= 0.0) or np.any(nir >= VIS_LO_EV):
        raise RuntimeError(
            f"{model_id} has near-IR peak predictions outside the near-IR region"
        )


def audit_selected_model_outputs(
    model_id: str,
    runs_root: Path,
    development_table_path: Path,
    development_split_path: Path,
    data_fingerprint_path: Path,
    output_dir: Path,
) -> dict:
    if model_id not in MODEL_SPECS:
        raise ValueError(f"selected-output audit requires a neural model: {model_id}")
    fingerprint = json.loads(data_fingerprint_path.read_text())
    if sha256_file(development_table_path) != fingerprint["development_table_sha256"]:
        raise RuntimeError("development table hash mismatch")
    if sha256_file(development_split_path) != fingerprint["development_split_sha256"]:
        raise RuntimeError("development split hash mismatch")
    split = json.loads(development_split_path.read_text())
    if "test" in split:
        raise RuntimeError("development split unexpectedly contains test IDs")
    cleaned = pd.read_parquet(development_table_path)
    expected_ids = set(split["train"]) | set(split["val"])
    if set(cleaned["id"]) != expected_ids:
        raise RuntimeError("development table IDs do not match split IDs")

    members = [
        _load_valid_member(
            runs_root / f"{model_id}_seed{seed}",
            model_id,
            seed,
        )
        for seed in FROZEN_SEEDS
    ]
    valid_members = [member for member in members if member["status"] == "valid"]
    if not valid_members:
        raise RuntimeError(f"{model_id} has no valid checkpoint to audit")
    arrays = _predict_neural(
        valid_members,
        cleaned,
        split["val"],
        object_sha256(fingerprint),
        dataset_tag="selected_validation_full_output",
    )
    _validate_prediction_arrays(
        arrays,
        model_id,
        len(valid_members),
        len(split["val"]),
    )
    _assert_neural_constraints(arrays, model_id)

    reproduction = []
    for member_index, member in enumerate(valid_members):
        with np.load(member["validation_predictions"]) as stored:
            stored_ids = stored["molecule_ids"].astype(str)
            stored_energy = np.stack(
                [
                    stored["gasphase_prediction"],
                    stored["acetone_prediction"],
                ],
                axis=1,
            )
        if not np.array_equal(stored_ids, np.asarray(split["val"], dtype=str)):
            raise RuntimeError(f"{model_id} stored validation IDs are misaligned")
        difference = np.abs(arrays["energy_prediction"][member_index] - stored_energy)
        max_abs_difference = float(difference.max())
        if not np.allclose(
            arrays["energy_prediction"][member_index],
            stored_energy,
            rtol=1e-5,
            atol=1e-5,
        ):
            raise RuntimeError(
                f"{model_id} seed {member['seed']} does not reproduce its "
                f"stored validation energies; max difference={max_abs_difference}"
            )
        reproduction.append(
            {
                "seed": member["seed"],
                "checkpoint_sha256": member["artifact_sha256"],
                "stored_validation_predictions_sha256": member[
                    "validation_predictions_sha256"
                ],
                "max_abs_energy_reproduction_difference_eV": max_abs_difference,
            }
        )

    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "selected_checkpoint_validation_outputs_only",
        "test_information_used": False,
        "model_id": model_id,
        "valid_seeds": [member["seed"] for member in valid_members],
        "failed_seeds": [
            member["seed"] for member in members if member["status"] == "failed"
        ],
        "n_validation_molecules": len(split["val"]),
        "data_fingerprint_sha256": object_sha256(fingerprint),
        "checks": {
            "all_prediction_shapes_exact": True,
            "all_predictions_finite": True,
            "all_probabilities_in_unit_interval": True,
            "all_state_energies_in_selected_checkpoint_interval": True,
            "all_regional_peak_energies_in_region": True,
            "all_nonnegative_heads_nonnegative": True,
            "stored_energy_predictions_reproduced": True,
        },
        "members": reproduction,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_id}.json"
    if output_path.exists():
        raise RuntimeError(f"selected-output audit already exists: {output_path}")
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[selected-output-audit] {model_id}: {len(valid_members)} valid members")
    return result
