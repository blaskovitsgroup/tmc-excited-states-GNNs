import json

import joblib
import numpy as np

from tmqmg_es.provenance import object_sha256, sha256_file
from tmqmg_es.train.run_audit import _audit_valid_run, audit_runs


def test_valid_run_audit_rehashes_and_recomputes_mae(tmp_path):
    run_dir = tmp_path / "xgboost_descriptors_seed2737188456"
    run_dir.mkdir()
    artifact = run_dir / "model.joblib"
    joblib.dump({"model": "test"}, artifact)
    ids = np.array(["A", "B"])
    target = np.ones((2, 2, 30), dtype=np.float32)
    prediction = target + 0.1
    predictions = run_dir / "validation_energy_predictions.npz"
    np.savez_compressed(
        predictions,
        molecule_ids=ids,
        gasphase_prediction=prediction[:, 0],
        acetone_prediction=prediction[:, 1],
        gasphase_target=target[:, 0],
        acetone_target=target[:, 1],
    )
    config = {"model_id": "xgboost_descriptors", "seed": 2737188456}
    report = {
        "status": "valid",
        "config": config,
        "config_sha256": object_sha256(config),
        "data_fingerprint_sha256": "data",
        "model_sha256": sha256_file(artifact),
        "validation_predictions_sha256": sha256_file(predictions),
        "final_validation": {"joint_molecule_mae_eV": 0.1},
    }
    (run_dir / "report.json").write_text(json.dumps(report))
    row, reference = _audit_valid_run(run_dir, "xgboost_descriptors", 2737188456, None)
    assert row["status"] == "valid"
    assert np.isclose(row["validation_joint_molecule_mae_eV"], 0.1)
    assert reference["molecule_ids"].tolist() == ["A", "B"]


def test_resolved_failed_seed_does_not_make_ensemble_preliminary(tmp_path, monkeypatch):
    from tmqmg_es.train import run_audit

    monkeypatch.setattr(run_audit, "MODEL_IDS", ("test_model",))
    monkeypatch.setattr(run_audit, "FROZEN_SEEDS", (1, 2))
    for seed in (1, 2):
        run_dir = tmp_path / f"test_model_seed{seed}"
        run_dir.mkdir()
    (tmp_path / "test_model_seed1" / "failure.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "exception_type": "FloatingPointError",
                "message": "frozen validation rule",
            }
        )
    )
    (tmp_path / "test_model_seed2" / "failure.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "exception_type": "FloatingPointError",
                "message": "frozen validation rule",
            }
        )
    )
    result = audit_runs(tmp_path)
    assert result["models"]["test_model"]["ensemble_is_preliminary"] is False
    assert result["counts"]["failed"] == 2
