"""One-time locked-test prediction and preregistered analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .. import config as C
from ..data.dataset import TMQMGDataset
from ..data.gs_features import build_feature_frame, feature_matrix, target_matrix
from ..models.baselines import XGBoostConfig, _validate as validate_xgboost
from ..provenance import (
    git_commit,
    git_is_dirty,
    object_sha256,
    sha256_file,
)
from . import metrics as M
from .ensemble_manifest import MIN_VALID_SEEDS, MODEL_IDS
from .trainer import TrainConfig, _validate_config, build_model, pick_device

BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260724
PRACTICAL_THRESHOLD_EV = 0.005
PREREGISTERED_FAMILY_SIZE = 3
CONFIRMATORY_COMPARISONS = (
    ("painn_dnatq_fusion", "painn_3d"),
    ("painn_dnatq_fusion", "xgboost_descriptors"),
)


def create_unlock_record(
    ensemble_manifest_path: Path,
    output_path: Path,
    data_fingerprint_path: Path,
    locked_ids_path: Path,
) -> dict:
    """Create the deliberate test-unlock record; this performs no evaluation."""
    if output_path.exists():
        raise RuntimeError(f"test has already been unlocked: {output_path}")
    if git_is_dirty(C.REPO_ROOT):
        raise RuntimeError("test unlock requires a clean Git worktree")
    ensemble = json.loads(ensemble_manifest_path.read_text())
    fingerprint = json.loads(data_fingerprint_path.read_text())
    locked = json.loads(locked_ids_path.read_text())
    if ensemble["data_fingerprint_sha256"] != object_sha256(fingerprint):
        raise RuntimeError("ensemble/data fingerprint mismatch")
    if fingerprint["locked_test_ids_sha256"] != sha256_file(locked_ids_path):
        raise RuntimeError("locked test ID hash mismatch")
    record = {
        "schema_version": 1,
        "unlocked_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_before_unlock_record": git_commit(C.REPO_ROOT),
        "protocol_sha256": sha256_file(C.PROTOCOL_PATH),
        "ensemble_manifest_sha256": sha256_file(ensemble_manifest_path),
        "data_fingerprint_sha256": object_sha256(fingerprint),
        "locked_test_ids_sha256": sha256_file(locked_ids_path),
        "n_test_molecules": int(locked["size"]),
        "authorized_scope": (
            "one automated confirmatory prediction campaign for every frozen "
            "model followed by the preregistered analysis"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"[test-unlock] record created without evaluating test labels: {output_path}")
    return record


def _verify_locked_inputs(
    ensemble_manifest_path: Path,
    unlock_path: Path,
    data_fingerprint_path: Path,
    locked_ids_path: Path,
    cleaned_table_path: Path,
) -> tuple[dict, dict, list[str], pd.DataFrame]:
    if git_is_dirty(C.REPO_ROOT):
        raise RuntimeError("locked prediction requires a clean Git worktree")
    ensemble = json.loads(ensemble_manifest_path.read_text())
    unlock = json.loads(unlock_path.read_text())
    fingerprint = json.loads(data_fingerprint_path.read_text())
    locked = json.loads(locked_ids_path.read_text())
    if sha256_file(ensemble_manifest_path) != unlock["ensemble_manifest_sha256"]:
        raise RuntimeError("ensemble manifest changed after test unlock")
    if object_sha256(fingerprint) != unlock["data_fingerprint_sha256"]:
        raise RuntimeError("data fingerprint changed after test unlock")
    if sha256_file(locked_ids_path) != unlock["locked_test_ids_sha256"]:
        raise RuntimeError("test IDs changed after test unlock")
    if sha256_file(cleaned_table_path) != fingerprint["cleaned_table_sha256"]:
        raise RuntimeError("cleaned table hash mismatch")
    test_ids = locked["test"]
    if len(test_ids) != locked["size"] or len(set(test_ids)) != len(test_ids):
        raise RuntimeError("locked test IDs are malformed")
    cleaned = pd.read_parquet(cleaned_table_path)
    return ensemble, fingerprint, test_ids, cleaned


def _valid_members(model_entry: dict) -> list[dict]:
    members = [
        member for member in model_entry["members"] if member["status"] == "valid"
    ]
    if not members:
        raise RuntimeError("model has no valid fixed seeds to evaluate")
    expected_eligibility = len(members) >= MIN_VALID_SEEDS
    if model_entry.get("confirmatory_eligible") is not expected_eligibility:
        raise RuntimeError("ensemble eligibility does not match valid seed count")
    return members


def _validate_prediction_arrays(
    arrays: dict[str, np.ndarray],
    model_id: str,
    n_members: int,
    n_molecules: int,
) -> None:
    expected = {
        "energy_prediction": (n_members, n_molecules, len(C.SOLVENTS), C.N_STATES),
        "energy_target": (n_molecules, len(C.SOLVENTS), C.N_STATES),
    }
    if model_id != "xgboost_descriptors":
        expected.update(
            {
                "logf_prediction": (
                    n_members,
                    n_molecules,
                    len(C.SOLVENTS),
                    C.N_STATES,
                ),
                "band_probability": (
                    n_members,
                    n_molecules,
                    len(C.SOLVENTS),
                    len(C.REGIONS),
                ),
                "peak_energy_prediction": (
                    n_members,
                    n_molecules,
                    len(C.SOLVENTS),
                    len(C.REGIONS),
                ),
                "peak_logf_prediction": (
                    n_members,
                    n_molecules,
                    len(C.SOLVENTS),
                    len(C.REGIONS),
                ),
                "peak_logwidth_prediction": (
                    n_members,
                    n_molecules,
                    len(C.SOLVENTS),
                    len(C.REGIONS),
                ),
                "ct_probability": (
                    n_members,
                    n_molecules,
                    len(C.SOLVENTS),
                    len(C.CT_CLASSES),
                ),
                "nto_prediction": (
                    n_members,
                    n_molecules,
                    len(C.SOLVENTS),
                    2,
                ),
                "logf_target": (
                    n_molecules,
                    len(C.SOLVENTS),
                    C.N_STATES,
                ),
                "band_target": (
                    n_molecules,
                    len(C.SOLVENTS),
                    len(C.REGIONS),
                ),
                "peak_energy_target": (
                    n_molecules,
                    len(C.SOLVENTS),
                    len(C.REGIONS),
                ),
                "peak_logf_target": (
                    n_molecules,
                    len(C.SOLVENTS),
                    len(C.REGIONS),
                ),
                "peak_logwidth_target": (
                    n_molecules,
                    len(C.SOLVENTS),
                    len(C.REGIONS),
                ),
                "peak_mask": (
                    n_molecules,
                    len(C.SOLVENTS),
                    len(C.REGIONS),
                ),
                "ct_target": (n_molecules, len(C.SOLVENTS)),
                "ct_mask": (n_molecules, len(C.SOLVENTS)),
                "nto_target": (n_molecules, len(C.SOLVENTS), 2),
                "nto_mask": (n_molecules, len(C.SOLVENTS)),
            }
        )
    missing = set(expected) - set(arrays)
    if missing:
        raise RuntimeError(f"{model_id} prediction arrays missing: {sorted(missing)}")
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise RuntimeError(f"{model_id} {key} shape {arrays[key].shape} != {shape}")
    prediction_keys = [key for key in expected if "prediction" in key]
    for key in prediction_keys:
        if not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"{model_id} has nonfinite values in {key}")
    for key in ("band_probability", "ct_probability", "nto_prediction"):
        if key in arrays and (np.any(arrays[key] < 0.0) or np.any(arrays[key] > 1.0)):
            raise RuntimeError(f"{model_id} has probabilities outside [0, 1] in {key}")


def _predict_xgboost(
    members: list[dict],
    cleaned: pd.DataFrame,
    test_ids: list[str],
) -> dict[str, np.ndarray]:
    features = build_feature_frame(cleaned)
    matrix, names = feature_matrix(features, test_ids)
    predictions = []
    for member in members:
        artifact = joblib.load(member["artifact"])
        if artifact["config_sha256"] != member.get(
            "serialized_config_sha256", member["config_sha256"]
        ):
            raise RuntimeError("XGBoost artifact/config hash mismatch")
        if object_sha256(artifact["cfg"]) != artifact["config_sha256"]:
            raise RuntimeError("XGBoost serialized configuration hash mismatch")
        validate_xgboost(XGBoostConfig(**artifact["cfg"]))
        if artifact["feature_names"] != names:
            raise RuntimeError("XGBoost descriptor schema mismatch")
        standardized = artifact["standardizer"].transform(matrix)
        by_solvent = [
            artifact["models"][solvent].predict(standardized) for solvent in C.SOLVENTS
        ]
        predictions.append(np.stack(by_solvent, axis=1))
    target = np.stack(
        [target_matrix(cleaned, test_ids, solvent) for solvent in C.SOLVENTS],
        axis=1,
    )
    return {
        "energy_prediction": np.stack(predictions).astype(np.float32),
        "energy_target": target.astype(np.float32),
    }


def _evaluation_dataset(
    cfg: TrainConfig,
    cleaned: pd.DataFrame,
    molecule_ids: list[str],
    fingerprint_hash: str,
    dataset_tag: str,
) -> TMQMGDataset:
    natqg_dir = None
    suffix = ""
    if cfg.regime in ("C", "G"):
        natqg_dir = Path(cfg.natqg_root) / f"{cfg.natqg_type}_graphs"
        suffix = f"_{cfg.natqg_type}"
    return TMQMGDataset(
        Path(cfg.cache_dir),
        cleaned=cleaned,
        ids=molecule_ids,
        cutoff=cfg.cutoff,
        tag=f"{dataset_tag}{suffix}",
        natqg_dir=natqg_dir,
        source_fingerprint=fingerprint_hash,
    )


@torch.no_grad()
def _predict_neural(
    members: list[dict],
    cleaned: pd.DataFrame,
    test_ids: list[str],
    fingerprint_hash: str,
    dataset_tag: str = "locked_test",
) -> dict[str, np.ndarray]:
    device = pick_device()
    models = []
    configs = []
    for member in members:
        checkpoint = torch.load(
            member["artifact"], map_location=device, weights_only=False
        )
        cfg = TrainConfig(**checkpoint["cfg"])
        if checkpoint["config_sha256"] != member.get(
            "serialized_config_sha256", member["config_sha256"]
        ):
            raise RuntimeError("checkpoint/config hash mismatch")
        if object_sha256(checkpoint["cfg"]) != checkpoint["config_sha256"]:
            raise RuntimeError("checkpoint serialized configuration hash mismatch")
        _validate_config(cfg)
        model = build_model(cfg).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)
        configs.append(cfg)
    structural_configs = [
        (
            cfg.model_id,
            cfg.backbone,
            cfg.regime,
            cfg.natqg_type,
            cfg.cutoff,
            cfg.hidden,
        )
        for cfg in configs
    ]
    if len(set(structural_configs)) != 1:
        raise RuntimeError("ensemble members do not share an architecture")
    dataset = _evaluation_dataset(
        configs[0],
        cleaned,
        test_ids,
        fingerprint_hash,
        f"{dataset_tag}_{configs[0].model_id}",
    )
    loader = DataLoader(dataset, batch_size=configs[0].batch_size, shuffle=False)

    accumulated: dict[str, list[np.ndarray]] = {
        "energy_prediction": [],
        "logf_prediction": [],
        "band_probability": [],
        "peak_energy_prediction": [],
        "peak_logf_prediction": [],
        "peak_logwidth_prediction": [],
        "ct_probability": [],
        "nto_prediction": [],
        "energy_target": [],
        "logf_target": [],
        "band_target": [],
        "peak_energy_target": [],
        "peak_logf_target": [],
        "peak_logwidth_target": [],
        "peak_mask": [],
        "ct_target": [],
        "ct_mask": [],
        "nto_target": [],
        "nto_mask": [],
    }
    ordered_ids: list[str] = []
    for batch in loader:
        ordered_ids.extend(batch.mol_id)
        batch = batch.to(device)
        member_outputs = [model(batch) for model in models]

        def member_stack(key: str) -> np.ndarray:
            return (
                torch.stack(
                    [
                        torch.stack(
                            [output[solvent][key] for solvent in C.SOLVENTS],
                            dim=1,
                        )
                        for output in member_outputs
                    ],
                    dim=0,
                )
                .cpu()
                .numpy()
            )

        accumulated["energy_prediction"].append(member_stack("E"))
        accumulated["logf_prediction"].append(member_stack("logf"))
        accumulated["band_probability"].append(
            1 / (1 + np.exp(-member_stack("band_logits")))
        )
        accumulated["peak_energy_prediction"].append(member_stack("peak_E"))
        accumulated["peak_logf_prediction"].append(member_stack("peak_f"))
        accumulated["peak_logwidth_prediction"].append(member_stack("sigma"))
        accumulated["ct_probability"].append(
            torch.softmax(torch.from_numpy(member_stack("ct_logits")), dim=-1).numpy()
        )
        accumulated["nto_prediction"].append(member_stack("mfrac"))

        accumulated["energy_target"].append(batch.E.cpu().numpy())
        accumulated["logf_target"].append(batch.logf.cpu().numpy())
        accumulated["band_target"].append(batch.band.cpu().numpy())
        accumulated["peak_energy_target"].append(batch.peak_E.cpu().numpy())
        accumulated["peak_logf_target"].append(batch.peak_f.cpu().numpy())
        accumulated["peak_logwidth_target"].append(
            torch.log1p(batch.sigma.clamp(min=0)).cpu().numpy()
        )
        accumulated["peak_mask"].append(batch.peak_mask.cpu().numpy())
        accumulated["ct_target"].append(batch.ct.cpu().numpy())
        accumulated["ct_mask"].append(batch.ct_mask.cpu().numpy())
        accumulated["nto_target"].append(batch.mfrac.cpu().numpy())
        accumulated["nto_mask"].append(batch.nto_mask.cpu().numpy())

    if ordered_ids != test_ids:
        raise RuntimeError("test graph order differs from locked ID order")
    member_keys = {
        "energy_prediction",
        "logf_prediction",
        "band_probability",
        "peak_energy_prediction",
        "peak_logf_prediction",
        "peak_logwidth_prediction",
        "ct_probability",
        "nto_prediction",
    }
    result = {}
    for key, chunks in accumulated.items():
        axis = 1 if key in member_keys else 0
        result[key] = np.concatenate(chunks, axis=axis).astype(np.float32)
    return result


def predict_locked_model(
    model_id: str,
    ensemble_manifest_path: Path,
    unlock_path: Path,
    data_fingerprint_path: Path,
    locked_ids_path: Path,
    cleaned_table_path: Path,
    output_dir: Path,
) -> dict:
    if model_id not in MODEL_IDS:
        raise ValueError(f"model is not frozen: {model_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{model_id}.npz"
    metadata_path = output_dir / f"{model_id}.metadata.json"
    if prediction_path.exists() or metadata_path.exists():
        raise RuntimeError(f"locked prediction already exists for {model_id}")
    ensemble, fingerprint, test_ids, cleaned = _verify_locked_inputs(
        ensemble_manifest_path,
        unlock_path,
        data_fingerprint_path,
        locked_ids_path,
        cleaned_table_path,
    )
    members = _valid_members(ensemble["models"][model_id])
    if model_id == "xgboost_descriptors":
        arrays = _predict_xgboost(members, cleaned, test_ids)
    else:
        arrays = _predict_neural(
            members,
            cleaned,
            test_ids,
            object_sha256(fingerprint),
        )
    _validate_prediction_arrays(arrays, model_id, len(members), len(test_ids))
    np.savez_compressed(
        prediction_path,
        molecule_ids=np.asarray(test_ids, dtype=str),
        member_seeds=np.asarray([member["seed"] for member in members]),
        **arrays,
    )
    metadata = {
        "model_id": model_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(C.REPO_ROOT),
        "ensemble_manifest_sha256": sha256_file(ensemble_manifest_path),
        "unlock_record_sha256": sha256_file(unlock_path),
        "prediction_sha256": sha256_file(prediction_path),
        "n_members": len(members),
        "member_seeds": [member["seed"] for member in members],
        "confirmatory_eligible": ensemble["models"][model_id]["confirmatory_eligible"],
        "metrics_computed": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[locked-predict] {model_id}: predictions saved; metrics not computed")
    return metadata


def _energy_summary(ensemble_prediction: np.ndarray, target: np.ndarray) -> dict:
    gas = M.energy_metrics(ensemble_prediction[:, 0], target[:, 0])
    acetone = M.energy_metrics(ensemble_prediction[:, 1], target[:, 1])
    return {
        "primary_joint_molecule_mae_eV": M.joint_molecule_energy_mae(
            ensemble_prediction, target
        ),
        "gasphase": gas,
        "acetone": acetone,
        "derived_state_energy_shift": M.solvent_shift_metrics(
            ensemble_prediction[:, 0],
            ensemble_prediction[:, 1],
            target[:, 0],
            target[:, 1],
        ),
    }


def _uncertainty_summary(
    member_prediction: np.ndarray,
    target: np.ndarray,
    variance_scale: float,
) -> dict:
    mean_prediction = np.asarray(member_prediction, dtype=float).mean(axis=0)
    raw_variance = np.asarray(member_prediction, dtype=float).var(axis=0)
    calibrated_variance = raw_variance * variance_scale

    def metrics(variance: np.ndarray) -> dict:
        standard_deviation = np.sqrt(np.clip(variance, 0.0, None))
        return {
            "coverage_68": M.coverage(
                mean_prediction, standard_deviation, target, z=1.0
            ),
            "coverage_95": M.coverage(
                mean_prediction, standard_deviation, target, z=1.96
            ),
            "gaussian_nll": M.nll_gaussian(mean_prediction, variance, target),
        }

    return {
        "unit": "state_energy_eV",
        "aggregation": "all_finite_molecule_solvent_state_targets",
        "member_variance_definition": "population_variance_across_fixed_seeds",
        "validation_fitted_variance_scale": variance_scale,
        "raw": metrics(raw_variance),
        "variance_scaled": metrics(calibrated_variance),
    }


def _neural_secondary(values: dict[str, np.ndarray]) -> dict:
    result = {"solvents": {}}
    ensemble_energy = values["energy_prediction"].mean(axis=0)
    ensemble_logf = values["logf_prediction"].mean(axis=0)
    ensemble_band = values["band_probability"].mean(axis=0)
    ensemble_peak_energy = values["peak_energy_prediction"].mean(axis=0)
    ensemble_peak_logf = values["peak_logf_prediction"].mean(axis=0)
    ensemble_peak_logwidth = values["peak_logwidth_prediction"].mean(axis=0)
    ensemble_ct = values["ct_probability"].mean(axis=0)
    ensemble_nto = values["nto_prediction"].mean(axis=0)
    for solvent_index, solvent in enumerate(C.SOLVENTS):
        solvent_result = {
            "oscillator": M.oscillator_metrics(
                ensemble_logf[:, solvent_index],
                values["logf_target"][:, solvent_index],
            ),
            "spectrum": M.spectrum_metrics(
                ensemble_energy[:, solvent_index],
                np.expm1(ensemble_logf[:, solvent_index]),
                values["energy_target"][:, solvent_index],
                np.expm1(values["logf_target"][:, solvent_index]),
            ),
            "bands": {},
            "peaks": {},
        }
        for region_index, region in enumerate(C.REGIONS):
            band_probability = ensemble_band[:, solvent_index, region_index]
            band_target = values["band_target"][:, solvent_index, region_index]
            solvent_result["bands"][region] = M.binary_classification_metrics(
                band_probability, band_target, threshold=0.5
            )
            true_mask = values["peak_mask"][:, solvent_index, region_index].astype(bool)
            gated_mask = true_mask & (band_probability >= 0.5)
            solvent_result["peaks"][region] = {
                "n_true": int(true_mask.sum()),
                "n_true_and_predicted_present": int(gated_mask.sum()),
                "gated_recall": (
                    float(gated_mask.sum() / true_mask.sum())
                    if true_mask.any()
                    else float("nan")
                ),
                "conditional_energy_mae_eV": M.mae(
                    ensemble_peak_energy[:, solvent_index, region_index],
                    values["peak_energy_target"][:, solvent_index, region_index],
                    true_mask,
                ),
                "conditional_logf_mae": M.mae(
                    ensemble_peak_logf[:, solvent_index, region_index],
                    values["peak_logf_target"][:, solvent_index, region_index],
                    true_mask,
                ),
                "conditional_logwidth_mae": M.mae(
                    ensemble_peak_logwidth[:, solvent_index, region_index],
                    values["peak_logwidth_target"][:, solvent_index, region_index],
                    true_mask,
                ),
            }
        ct_mask = values["ct_mask"][:, solvent_index].astype(bool)
        ct_true_index = values["ct_target"][:, solvent_index]
        ct_pred_index = ensemble_ct[:, solvent_index].argmax(axis=-1)
        masked_ct_index = np.asarray(ct_true_index[ct_mask])
        if not np.isfinite(masked_ct_index).all():
            raise ValueError(f"{solvent} CT targets contain nonfinite class indices")
        rounded_ct_index = np.rint(masked_ct_index)
        if not np.array_equal(masked_ct_index, rounded_ct_index):
            raise ValueError(f"{solvent} CT targets contain noninteger class indices")
        masked_ct_index = rounded_ct_index.astype(np.int64)
        if (
            (masked_ct_index < 0).any()
            or (masked_ct_index >= len(C.CT_CLASSES)).any()
        ):
            raise ValueError(f"{solvent} CT targets contain out-of-range class indices")
        ct_true = np.full(len(ct_true_index), None, dtype=object)
        ct_pred = np.full(len(ct_pred_index), None, dtype=object)
        ct_true[ct_mask] = np.asarray(C.CT_CLASSES)[masked_ct_index]
        ct_pred[ct_mask] = np.asarray(C.CT_CLASSES)[ct_pred_index[ct_mask]]
        solvent_result["ct"] = M.classification_metrics(
            ct_pred, ct_true, list(C.CT_CLASSES)
        )
        nto_mask = values["nto_mask"][:, solvent_index].astype(bool)
        solvent_result["nto"] = {
            "n": int(nto_mask.sum()),
            "metal_fraction_mae": M.mae(
                ensemble_nto[:, solvent_index],
                values["nto_target"][:, solvent_index],
                nto_mask[:, None].repeat(2, axis=1),
            ),
        }
        result["solvents"][solvent] = solvent_result
    return result


def _bootstrap_difference(
    error_a: np.ndarray,
    error_b: np.ndarray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    difference = np.asarray(error_a) - np.asarray(error_b)
    if not np.isfinite(difference).all():
        raise ValueError("primary molecule errors contain nonfinite values")
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples, dtype=float)
    chunk = 200
    for start in range(0, resamples, chunk):
        stop = min(start + chunk, resamples)
        indices = rng.integers(0, len(difference), size=(stop - start, len(difference)))
        boot[start:stop] = difference[indices].mean(axis=1)
    lower, upper = np.quantile(boot, [0.025, 0.975])
    left = (np.sum(boot <= 0) + 1) / (resamples + 1)
    right = (np.sum(boot >= 0) + 1) / (resamples + 1)
    return {
        "mean_A_minus_B_eV": float(difference.mean()),
        "mae_improvement_B_minus_A_eV": float(-difference.mean()),
        "ci95_A_minus_B_eV": [float(lower), float(upper)],
        "paired_bootstrap_p_raw": float(min(1.0, 2 * min(left, right))),
        "resamples": resamples,
        "seed": seed,
    }


def _holm_adjust(raw_p: list[float]) -> list[float]:
    order = np.argsort(raw_p)
    adjusted = np.empty(len(raw_p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(raw_p) - rank) * raw_p[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _analyze_confirmatory_comparisons(
    primary_errors: dict[str, np.ndarray],
    model_eligibility: dict[str, bool],
) -> list[dict]:
    comparisons = []
    multiplicity_p = []
    for model_a, model_b in CONFIRMATORY_COMPARISONS:
        ineligible = [
            model_id
            for model_id in (model_a, model_b)
            if not model_eligibility[model_id]
        ]
        if ineligible:
            comparison = {
                "model_A": model_a,
                "model_B": model_b,
                "status": "not_tested",
                "reason": "fewer_than_four_valid_fixed_seeds",
                "ineligible_models": ineligible,
                "paired_bootstrap_p_raw": None,
                "multiplicity_p_value": 1.0,
            }
        else:
            comparison = _bootstrap_difference(
                primary_errors[model_a], primary_errors[model_b]
            )
            comparison.update(
                {
                    "model_A": model_a,
                    "model_B": model_b,
                    "status": "tested",
                    "multiplicity_p_value": comparison["paired_bootstrap_p_raw"],
                }
            )
        multiplicity_p.append(comparison["multiplicity_p_value"])
        comparisons.append(comparison)

    # Retain the original multiplicity correction when exporting a model subset.
    padded_p = multiplicity_p + [1.0] * (PREREGISTERED_FAMILY_SIZE - len(multiplicity_p))
    for comparison, adjusted_p in zip(comparisons, _holm_adjust(padded_p)):
        comparison["holm_adjusted_p"] = adjusted_p
        if comparison["status"] == "not_tested":
            comparison["decision"] = "not_tested_ineligible_ensemble"
            continue
        improvement = comparison["mae_improvement_B_minus_A_eV"]
        _, upper = comparison["ci95_A_minus_B_eV"]
        if abs(improvement) < PRACTICAL_THRESHOLD_EV:
            decision = "practical_tie"
        elif improvement >= PRACTICAL_THRESHOLD_EV and upper < 0 and adjusted_p < 0.05:
            decision = "A_outperforms_B"
        else:
            decision = "inconclusive"
        comparison["decision"] = decision
    return comparisons


def summarize_locked_predictions(
    prediction_dir: Path,
    ensemble_manifest_path: Path,
    unlock_path: Path,
    output_dir: Path,
) -> dict:
    ledger_path = output_dir / "locked_evaluation_ledger.json"
    if ledger_path.exists():
        raise RuntimeError("confirmatory test metrics have already been summarized")
    ensemble_manifest = json.loads(ensemble_manifest_path.read_text())
    unlock = json.loads(unlock_path.read_text())
    ensemble_manifest_hash = sha256_file(ensemble_manifest_path)
    unlock_hash = sha256_file(unlock_path)
    if unlock["ensemble_manifest_sha256"] != ensemble_manifest_hash:
        raise RuntimeError("unlock record does not match ensemble manifest")
    prediction_hashes = {}
    prediction_metadata_hashes = {}
    molecule_ids = None
    energy_target = None
    results = {"models": {}}
    primary_errors = {}
    for model_id in MODEL_IDS:
        path = prediction_dir / f"{model_id}.npz"
        metadata_path = prediction_dir / f"{model_id}.metadata.json"
        metadata = json.loads(metadata_path.read_text())
        model_entry = ensemble_manifest["models"][model_id]
        expected_members = [
            member["seed"]
            for member in model_entry["members"]
            if member["status"] == "valid"
        ]
        if metadata["model_id"] != model_id:
            raise RuntimeError(f"prediction metadata identity mismatch for {model_id}")
        if metadata["ensemble_manifest_sha256"] != ensemble_manifest_hash:
            raise RuntimeError(f"prediction manifest mismatch for {model_id}")
        if metadata["unlock_record_sha256"] != unlock_hash:
            raise RuntimeError(f"prediction unlock mismatch for {model_id}")
        if metadata["member_seeds"] != expected_members:
            raise RuntimeError(f"prediction member mismatch for {model_id}")
        if metadata["n_members"] != len(expected_members):
            raise RuntimeError(f"prediction member count mismatch for {model_id}")
        if (
            metadata["confirmatory_eligible"]
            is not model_entry["confirmatory_eligible"]
        ):
            raise RuntimeError(f"prediction eligibility mismatch for {model_id}")
        if metadata["metrics_computed"] is not False:
            raise RuntimeError(f"prediction metadata is not pre-metric for {model_id}")
        if sha256_file(path) != metadata["prediction_sha256"]:
            raise RuntimeError(f"prediction hash mismatch for {model_id}")
        values = dict(np.load(path))
        if values["member_seeds"].tolist() != expected_members:
            raise RuntimeError(f"prediction seed array mismatch for {model_id}")
        ids = values["molecule_ids"].astype(str)
        _validate_prediction_arrays(values, model_id, len(expected_members), len(ids))
        if molecule_ids is None:
            molecule_ids = ids
            energy_target = values["energy_target"]
        elif not np.array_equal(ids, molecule_ids) or not np.allclose(
            values["energy_target"], energy_target, equal_nan=True
        ):
            raise RuntimeError(f"test alignment mismatch for {model_id}")
        ensemble_energy = values["energy_prediction"].mean(axis=0)
        energy = _energy_summary(ensemble_energy, values["energy_target"])
        eligibility = ensemble_manifest["models"][model_id]["confirmatory_eligible"]
        model_result = {
            "confirmatory_eligible": eligibility,
            "n_valid_fixed_seeds": int(len(values["member_seeds"])),
            "energy": energy,
            "uncertainty": _uncertainty_summary(
                values["energy_prediction"],
                values["energy_target"],
                model_entry["validation_calibration"]["epistemic_variance_scale"],
            ),
        }
        if model_id != "xgboost_descriptors":
            model_result["secondary"] = _neural_secondary(values)
        results["models"][model_id] = model_result
        primary_errors[model_id] = M.molecule_energy_errors(
            ensemble_energy, values["energy_target"]
        )
        prediction_hashes[model_id] = sha256_file(path)
        prediction_metadata_hashes[model_id] = sha256_file(metadata_path)

    model_eligibility = {
        model_id: model["confirmatory_eligible"]
        for model_id, model in ensemble_manifest["models"].items()
    }
    comparison_results = _analyze_confirmatory_comparisons(
        primary_errors, model_eligibility
    )
    results["confirmatory_comparisons"] = comparison_results
    results["decision_rule"] = {
        "minimum_improvement_eV": PRACTICAL_THRESHOLD_EV,
        "ci_must_exclude_zero": True,
        "holm_adjusted_alpha": 0.05,
        "preregistered_family_size": PREREGISTERED_FAMILY_SIZE,
        "ineligible_comparison_multiplicity_p_value": 1.0,
    }
    results = _json_safe(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "locked_test_results.json"
    results_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    per_molecule = pd.DataFrame({"id": molecule_ids})
    for model_id, errors in primary_errors.items():
        per_molecule[f"{model_id}_joint_energy_mae_eV"] = errors
    per_molecule_path = output_dir / "locked_test_per_molecule_errors.csv"
    per_molecule.to_csv(per_molecule_path, index=False)

    ledger = {
        "schema_version": 1,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(C.REPO_ROOT),
        "protocol_sha256": sha256_file(C.PROTOCOL_PATH),
        "ensemble_manifest_sha256": sha256_file(ensemble_manifest_path),
        "unlock_record_sha256": sha256_file(unlock_path),
        "prediction_sha256": prediction_hashes,
        "prediction_metadata_sha256": prediction_metadata_hashes,
        "results_sha256": sha256_file(results_path),
        "per_molecule_errors_sha256": sha256_file(per_molecule_path),
        "n_test_molecules": int(len(molecule_ids)),
        "confirmatory_evaluation_number": 1,
        "models_evaluated": list(MODEL_IDS),
        "ensemble_manifest_content_sha256": ensemble_manifest["content_sha256"],
    }
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    print(f"[locked-summary] confirmatory evaluation complete -> {results_path}")
    return results
