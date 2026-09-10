"""Audit partially or fully retrieved training-and-validation run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..provenance import sha256_file
from .ensemble_manifest import MODEL_IDS, _artifact_name
from .metrics import joint_molecule_energy_mae
from .trainer import FROZEN_SEEDS


def _validation_mae(report: dict) -> float | None:
    final = report.get("final_validation", {})
    value = final.get("joint_molecule_mae_eV")
    if value is None:
        value = report.get("best_val_joint_molecule_mae_eV")
    return None if value is None else float(value)


def _audit_valid_run(
    run_dir: Path,
    model_id: str,
    seed: int,
    reference: dict | None,
) -> tuple[dict, dict]:
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("status") != "valid":
        raise RuntimeError(f"non-valid report: {report_path}")
    config = report["config"]
    if config["model_id"] != model_id or int(config["seed"]) != seed:
        raise RuntimeError(f"run identity mismatch: {run_dir}")

    artifact = run_dir / _artifact_name(model_id)
    predictions_path = run_dir / "validation_energy_predictions.npz"
    artifact_hash_key = (
        "model_sha256" if model_id == "xgboost_descriptors" else "checkpoint_sha256"
    )
    checks = {
        "artifact_exists": artifact.exists(),
        "predictions_exist": predictions_path.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"required artifact missing: {run_dir}")
    checks["artifact_hash_matches"] = sha256_file(artifact) == report[artifact_hash_key]
    checks["prediction_hash_matches"] = (
        sha256_file(predictions_path) == report["validation_predictions_sha256"]
    )
    if not all(checks.values()):
        raise RuntimeError(f"artifact hash mismatch: {run_dir}")

    with np.load(predictions_path) as values:
        molecule_ids = values["molecule_ids"].astype(str)
        target = np.stack([values["gasphase_target"], values["acetone_target"]], axis=1)
        prediction = np.stack(
            [values["gasphase_prediction"], values["acetone_prediction"]], axis=1
        )
    if len(molecule_ids) != len(set(molecule_ids)):
        raise RuntimeError(f"duplicate validation molecule IDs: {predictions_path}")
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"nonfinite selected validation prediction: {run_dir}")
    if np.any(prediction <= 0) or np.any(prediction > 15):
        raise RuntimeError(f"selected validation prediction outside (0, 15]: {run_dir}")

    if reference is None:
        reference = {"molecule_ids": molecule_ids, "target": target}
    elif not np.array_equal(reference["molecule_ids"], molecule_ids) or not np.allclose(
        reference["target"], target, equal_nan=True
    ):
        raise RuntimeError(f"validation alignment mismatch: {run_dir}")

    measured_mae = joint_molecule_energy_mae(prediction, target)
    reported_mae = _validation_mae(report)
    if reported_mae is None or not np.isclose(measured_mae, reported_mae, atol=1e-7):
        raise RuntimeError(f"validation MAE mismatch: {run_dir}")
    return (
        {
            "model_id": model_id,
            "seed": seed,
            "status": "valid",
            "best_epoch": report.get("best_epoch"),
            "validation_joint_molecule_mae_eV": measured_mae,
            "artifact": str(artifact),
            "artifact_sha256": sha256_file(artifact),
            "report_sha256": sha256_file(report_path),
            "validation_predictions": str(predictions_path),
            "validation_predictions_sha256": sha256_file(predictions_path),
            "config_sha256": report["config_sha256"],
            "data_fingerprint_sha256": report["data_fingerprint_sha256"],
            "checks": checks,
        },
        reference,
    )


def audit_runs(runs_root: Path) -> dict:
    rows = []
    reference = None
    valid_predictions: dict[str, list[np.ndarray]] = {}
    valid_target = None
    data_fingerprints = set()
    for model_id in MODEL_IDS:
        for seed in FROZEN_SEEDS:
            run_dir = runs_root / f"{model_id}_seed{seed}"
            report_path = run_dir / "report.json"
            failure_path = run_dir / "failure.json"
            if report_path.exists():
                row, reference = _audit_valid_run(run_dir, model_id, seed, reference)
                rows.append(row)
                data_fingerprints.add(row["data_fingerprint_sha256"])
                with np.load(row["validation_predictions"]) as values:
                    prediction = np.stack(
                        [
                            values["gasphase_prediction"],
                            values["acetone_prediction"],
                        ],
                        axis=1,
                    )
                    valid_target = np.stack(
                        [values["gasphase_target"], values["acetone_target"]], axis=1
                    )
                valid_predictions.setdefault(model_id, []).append(prediction)
            elif failure_path.exists():
                failure = json.loads(failure_path.read_text())
                rows.append(
                    {
                        "model_id": model_id,
                        "seed": seed,
                        "status": "failed",
                        "failure_type": failure.get("exception_type"),
                        "failure_message": failure.get("message"),
                        "failure_sha256": sha256_file(failure_path),
                    }
                )
            elif run_dir.exists():
                rows.append(
                    {"model_id": model_id, "seed": seed, "status": "incomplete"}
                )
            else:
                rows.append({"model_id": model_id, "seed": seed, "status": "missing"})

    if len(data_fingerprints) > 1:
        raise RuntimeError("valid runs use different data fingerprints")
    models = {}
    for model_id in MODEL_IDS:
        members = [row for row in rows if row["model_id"] == model_id]
        predictions = valid_predictions.get(model_id, [])
        unresolved = any(row["status"] in {"incomplete", "missing"} for row in members)
        models[model_id] = {
            "valid": sum(row["status"] == "valid" for row in members),
            "failed": sum(row["status"] == "failed" for row in members),
            "incomplete": sum(row["status"] == "incomplete" for row in members),
            "missing": sum(row["status"] == "missing" for row in members),
            "validation_ensemble_joint_molecule_mae_eV": (
                joint_molecule_energy_mae(np.mean(predictions, axis=0), valid_target)
                if predictions
                else None
            ),
            "ensemble_is_preliminary": unresolved,
            "confirmatory_eligible": len(predictions) >= 4 and not unresolved,
        }
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("valid", "failed", "incomplete", "missing")
    }
    return {
        "schema_version": 2,
        "scope": "training_and_validation_only",
        "test_information_used": False,
        "expected_runs": len(MODEL_IDS) * len(FROZEN_SEEDS),
        "counts": counts,
        "data_fingerprint_sha256": (
            next(iter(data_fingerprints)) if data_fingerprints else None
        ),
        "models": models,
        "runs": rows,
    }
