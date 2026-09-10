import numpy as np
import yaml
from tmqmg_es.provenance import sha256_file

from tmqmg_es import config as C
from tmqmg_es.models.baselines import XGBoostConfig, _validate as validate_xgboost
from tmqmg_es.models.heads import PeakHead, VIS_HI_EV, VIS_LO_EV
from tmqmg_es.train.locked_eval import (
    _analyze_confirmatory_comparisons,
    _bootstrap_difference,
    _holm_adjust,
    _json_safe,
    _uncertainty_summary,
    _valid_members,
    _validate_prediction_arrays,
)
from tmqmg_es.train.selected_output_audit import _assert_neural_constraints
from tmqmg_es.train.trainer import (
    FROZEN_SEEDS,
    MODEL_SPECS,
    POST_CONFIRMATORY_RERUN_MODELS,
    TrainConfig,
    _optimizer_parameter_groups,
    _set_epoch_optimization_state,
    _validate_config,
    _validation_prediction_is_finite,
    _validation_prediction_is_valid,
)


def test_every_frozen_neural_model_has_a_valid_config():
    config_dir = C.REPO_ROOT / "configs/models"
    for model_id in MODEL_SPECS:
        values = yaml.safe_load((config_dir / f"{model_id}.yaml").read_text())
        cfg = TrainConfig(**values)
        assert cfg.model_id == model_id
        _validate_config(cfg)


def test_xgboost_config_matches_protocol():
    values = yaml.safe_load(
        (C.REPO_ROOT / "configs/models/xgboost_descriptors.yaml").read_text()
    )
    cfg = XGBoostConfig(**values)
    assert cfg.seed == FROZEN_SEEDS[0]
    validate_xgboost(cfg)


def test_confirmatory_config_rejects_seed_replacement():
    values = yaml.safe_load((C.REPO_ROOT / "configs/models/painn_3d.yaml").read_text())
    values["seed"] = 11
    with np.testing.assert_raises(ValueError):
        _validate_config(TrainConfig(**values))


def test_confirmatory_config_rejects_followup_optimization_options():
    values = yaml.safe_load((C.REPO_ROOT / "configs/models/painn_3d.yaml").read_text())
    values["backbone_lr"] = 1e-5
    with np.testing.assert_raises(ValueError):
        _validate_config(TrainConfig(**values))




def test_current_rerun_configs_use_main_project_status():
    config_dir = C.REPO_ROOT / "configs/models"
    for model_id in POST_CONFIRMATORY_RERUN_MODELS:
        values = yaml.safe_load((config_dir / f"{model_id}.yaml").read_text())
        cfg = TrainConfig(**values)
        _validate_config(cfg)
        assert cfg.confirmatory is False
        assert cfg.study_status == "post_confirmatory_rerun"


def test_post_confirmatory_rerun_rejects_unaffected_model():
    values = yaml.safe_load(
        (C.REPO_ROOT / "configs/models/painn_3d.yaml").read_text()
    )
    values.update(confirmatory=False, study_status="post_confirmatory_rerun")
    with np.testing.assert_raises(ValueError):
        _validate_config(TrainConfig(**values))




def test_energy_range_rule_applies_to_the_selected_checkpoint():
    transient = {
        "prediction": {
            "gasphase": np.array([[0.2, 16.0]]),
            "acetone": np.array([[0.3, 14.0]]),
        }
    }
    assert _validation_prediction_is_finite(transient) == (True, None)
    valid, reason = _validation_prediction_is_valid(transient)
    assert valid is False
    assert "outside (0, 15.0]" in reason


def test_peak_energy_head_stays_inside_regions():
    torch = __import__("torch")
    head = PeakHead(8)
    prediction = head(torch.randn(128, 8))["peak_E"]
    assert bool((prediction[:, 0] > VIS_HI_EV).all())
    assert bool(
        ((prediction[:, 1] >= VIS_LO_EV) & (prediction[:, 1] <= VIS_HI_EV)).all()
    )
    assert bool(((prediction[:, 2] > 0) & (prediction[:, 2] < VIS_LO_EV)).all())


def test_bootstrap_and_holm_rules_are_deterministic():
    error_a = np.linspace(0.10, 0.20, 200)
    error_b = error_a + 0.01
    first = _bootstrap_difference(error_a, error_b, resamples=1000, seed=7)
    second = _bootstrap_difference(error_a, error_b, resamples=1000, seed=7)
    assert first == second
    assert first["mae_improvement_B_minus_A_eV"] > 0.009
    assert first["ci95_A_minus_B_eV"][1] < 0
    adjusted = _holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]


def test_ineligible_model_is_evaluated_but_not_used_for_confirmatory_claim():
    members = [{"seed": seed, "status": "valid"} for seed in FROZEN_SEEDS[:3]]
    model_entry = {"members": members, "confirmatory_eligible": False}
    assert _valid_members(model_entry) == members




def test_uncertainty_summary_applies_validation_variance_scale():
    target = np.zeros((2, 2, 1))
    members = np.array(
        [
            [[[-1.0], [-1.0]], [[-1.0], [-1.0]]],
            [[[1.0], [1.0]], [[1.0], [1.0]]],
        ]
    )
    summary = _uncertainty_summary(members, target, variance_scale=4.0)
    assert summary["raw"]["coverage_68"] == 1.0
    assert summary["variance_scaled"]["coverage_95"] == 1.0
    assert summary["validation_fitted_variance_scale"] == 4.0


def test_prediction_array_validation_rejects_nonfinite_predictions():
    arrays = {
        "energy_prediction": np.ones((2, 3, 2, 30)),
        "energy_target": np.ones((3, 2, 30)),
    }
    _validate_prediction_arrays(arrays, "xgboost_descriptors", 2, 3)
    arrays["energy_prediction"][0, 0, 0, 0] = np.nan
    with np.testing.assert_raises(RuntimeError):
        _validate_prediction_arrays(arrays, "xgboost_descriptors", 2, 3)


def test_selected_output_constraints_reject_wrong_visible_peak_region():
    arrays = {
        "energy_prediction": np.ones((2, 3, 2, 30)),
        "logf_prediction": np.ones((2, 3, 2, 30)),
        "peak_energy_prediction": np.zeros((2, 3, 2, 3)),
        "peak_logf_prediction": np.ones((2, 3, 2, 3)),
        "peak_logwidth_prediction": np.ones((2, 3, 2, 3)),
    }
    arrays["peak_energy_prediction"][..., 0] = VIS_HI_EV + 0.1
    arrays["peak_energy_prediction"][..., 1] = VIS_LO_EV - 0.1
    arrays["peak_energy_prediction"][..., 2] = VIS_LO_EV - 0.1
    with np.testing.assert_raises(RuntimeError):
        _assert_neural_constraints(arrays, "test_model")


def test_json_safe_converts_nonfinite_metrics_to_null_values():
    assert _json_safe({"a": np.nan, "b": [np.float64(np.inf), 1.0]}) == {
        "a": None,
        "b": [None, 1.0],
    }


def test_training_config_has_no_locked_test_path():
    values = yaml.safe_load((C.REPO_ROOT / "configs/models/painn_3d.yaml").read_text())
    serialized = str(values).lower()
    assert "locked" not in serialized
    assert "test" not in serialized


def test_cluster_prepare_uses_a_configurable_natqg_root():
    script = (C.REPO_ROOT / "scripts/cluster/prepare.sbatch").read_text()
    assert "TMQMG_SOURCE_ROOT" in script
    assert '"$SOURCE/natqg"' in script
    assert "--natqg-type" in script
    assert "--natq-type" not in script
    assert "--natqg-root" in script
    assert "--natq-root" not in script


def test_cluster_jobs_use_a_configurable_environment():
    for script in (C.REPO_ROOT / "scripts/cluster").glob("*.sbatch"):
        text = script.read_text()
        assert "TMQMG_PYTHON" in text
        assert "TMQMG_REPO_ROOT" in text
        assert "/scratch/ismael" not in text




def test_protocol_excludes_direct_shift_training():
    protocol = yaml.safe_load(C.PROTOCOL_PATH.read_text())
    assert protocol["targets"]["direct_lambda_shift"]["enabled"] is False
    assert protocol["targets"]["direct_intensity_shift"]["enabled"] is False
    assert protocol["loss_weights"]["direct_shift"] == 0.0
