"""Freeze ensemble membership and validation-only calibration before test unlock."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import config as C
from ..provenance import git_commit, git_is_dirty, object_sha256, sha256_file
from .metrics import joint_molecule_energy_mae
from .trainer import FROZEN_SEEDS, MODEL_SPECS

MODEL_IDS = tuple(MODEL_SPECS) + ("xgboost_descriptors",)
MIN_VALID_SEEDS = 4


def _artifact_name(model_id: str) -> str:
    return "model.joblib" if model_id == "xgboost_descriptors" else "best.pt"


def _load_valid_member(run_dir: Path, model_id: str, seed: int) -> dict:
    report_path = run_dir / "report.json"
    if not report_path.exists():
        failure_path = run_dir / "failure.json"
        if failure_path.exists():
            failure = json.loads(failure_path.read_text())
            return {
                "seed": seed,
                "status": "failed",
                "failure_sha256": sha256_file(failure_path),
                "failure": failure,
            }
        raise RuntimeError(f"run is incomplete: {run_dir}")
    report = json.loads(report_path.read_text())
    if report.get("status") != "valid":
        raise RuntimeError(f"run report is not valid: {report_path}")
    config = report["config"]
    if config["model_id"] != model_id or int(config["seed"]) != seed:
        raise RuntimeError(f"run identity mismatch: {run_dir}")
    if object_sha256(config) != report["config_sha256"]:
        raise RuntimeError(f"run configuration hash mismatch: {run_dir}")
    artifact = run_dir / _artifact_name(model_id)
    predictions = run_dir / "validation_energy_predictions.npz"
    if not artifact.exists() or not predictions.exists():
        raise RuntimeError(f"run lacks required artifacts: {run_dir}")
    artifact_hash_key = (
        "model_sha256" if model_id == "xgboost_descriptors" else "checkpoint_sha256"
    )
    if sha256_file(artifact) != report[artifact_hash_key]:
        raise RuntimeError(f"model artifact hash mismatch: {artifact}")
    if sha256_file(predictions) != report["validation_predictions_sha256"]:
        raise RuntimeError(f"validation prediction hash mismatch: {predictions}")
    return {
        "seed": seed,
        "status": "valid",
        "run_dir": str(run_dir),
        "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "report_sha256": sha256_file(report_path),
        "validation_predictions": str(predictions),
        "validation_predictions_sha256": sha256_file(predictions),
        "config_sha256": report["config_sha256"],
        "serialized_config_sha256": report.get(
            "serialized_config_sha256", report["config_sha256"]
        ),
        "data_fingerprint_sha256": report["data_fingerprint_sha256"],
    }


def _validation_calibration(valid_members: list[dict]) -> dict:
    predictions = []
    molecule_ids = None
    target = None
    for member in valid_members:
        values = np.load(member["validation_predictions"])
        ids = values["molecule_ids"].astype(str)
        member_target = np.stack(
            [values["gasphase_target"], values["acetone_target"]], axis=1
        )
        member_prediction = np.stack(
            [
                values["gasphase_prediction"],
                values["acetone_prediction"],
            ],
            axis=1,
        )
        if molecule_ids is None:
            molecule_ids = ids
            target = member_target
        elif not np.array_equal(ids, molecule_ids) or not np.allclose(
            member_target, target, equal_nan=True
        ):
            raise RuntimeError("validation predictions are not identically aligned")
        predictions.append(member_prediction)
    stacked = np.stack(predictions)
    mean_prediction = stacked.mean(axis=0)
    variance = stacked.var(axis=0)
    valid = np.isfinite(mean_prediction) & np.isfinite(target) & np.isfinite(variance)
    scale = float(
        np.mean(
            (mean_prediction[valid] - target[valid]) ** 2
            / np.clip(variance[valid], 1e-8, None)
        )
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("validation variance scale is not finite and positive")
    return {
        "n_validation_molecules": int(len(molecule_ids)),
        "ensemble_joint_molecule_mae_eV": joint_molecule_energy_mae(
            mean_prediction, target
        ),
        "epistemic_variance_scale": scale,
        "definition": "mean squared residual divided by ensemble variance",
    }


def freeze_ensemble_manifest(
    runs_root: Path = C.REPO_ROOT / "outputs/runs",
    output_path: Path = C.REPO_ROOT / "manifests/final_ensemble_manifest.json",
) -> dict:
    if git_is_dirty(C.REPO_ROOT):
        raise RuntimeError("ensemble freeze requires a clean Git worktree")
    models = {}
    data_fingerprint = None
    for model_id in MODEL_IDS:
        members = [
            _load_valid_member(runs_root / f"{model_id}_seed{seed}", model_id, seed)
            for seed in FROZEN_SEEDS
        ]
        valid_members = [member for member in members if member["status"] == "valid"]
        if not valid_members:
            raise RuntimeError(f"{model_id} has no valid fixed seeds to evaluate")
        confirmatory_eligible = len(valid_members) >= MIN_VALID_SEEDS
        member_fingerprints = {
            member["data_fingerprint_sha256"] for member in valid_members
        }
        if len(member_fingerprints) != 1:
            raise RuntimeError(f"{model_id} members used different data")
        current_fingerprint = next(iter(member_fingerprints))
        if data_fingerprint is None:
            data_fingerprint = current_fingerprint
        elif current_fingerprint != data_fingerprint:
            raise RuntimeError("model families used different data fingerprints")
        models[model_id] = {
            "members": members,
            "valid_seeds": [member["seed"] for member in valid_members],
            "aggregation": "arithmetic_mean_of_all_valid_fixed_seeds",
            "confirmatory_eligible": confirmatory_eligible,
            "minimum_valid_seeds_for_confirmatory_claim": MIN_VALID_SEEDS,
            "eligibility_reason": (
                None
                if confirmatory_eligible
                else (
                    f"{len(valid_members)} valid fixed seeds; "
                    f"{MIN_VALID_SEEDS} required"
                )
            ),
            "validation_calibration": _validation_calibration(valid_members),
        }

    eligible_models = [
        model_id for model_id, model in models.items() if model["confirmatory_eligible"]
    ]
    manifest = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(C.REPO_ROOT),
        "protocol_sha256": sha256_file(C.PROTOCOL_PATH),
        "data_fingerprint_sha256": data_fingerprint,
        "fixed_seeds": list(FROZEN_SEEDS),
        "models": models,
        "confirmatory_eligible_models": eligible_models,
        "descriptive_only_models": [
            model_id for model_id in models if model_id not in eligible_models
        ],
        "test_information_used": False,
    }
    manifest["content_sha256"] = object_sha256(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[ensemble-freeze] {len(models)} model ensembles -> {output_path}")
    return manifest
