import json

import pytest

from tmqmg_es.provenance import object_sha256, sha256_file
from tmqmg_es.train.ensemble_manifest import _load_valid_member


def test_packaged_config_retains_serialized_identity(tmp_path):
    config = {"model_id": "painn_3d", "seed": 2737188456}
    checkpoint = tmp_path / "best.pt"
    predictions = tmp_path / "validation_energy_predictions.npz"
    checkpoint.write_bytes(b"unchanged checkpoint")
    predictions.write_bytes(b"unchanged predictions")
    report = {
        "status": "valid",
        "config": config,
        "config_sha256": object_sha256(config),
        "serialized_config_sha256": "a" * 64,
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation_predictions_sha256": sha256_file(predictions),
        "data_fingerprint_sha256": "b" * 64,
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    member = _load_valid_member(tmp_path, "painn_3d", 2737188456)
    assert member["config_sha256"] == object_sha256(config)
    assert member["serialized_config_sha256"] == "a" * 64

    report["config"]["hidden"] = 999
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="configuration hash mismatch"):
        _load_valid_member(tmp_path, "painn_3d", 2737188456)
