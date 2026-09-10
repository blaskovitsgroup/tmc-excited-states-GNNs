"""Validation-only XGBoost training for the frozen descriptor baseline."""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from .. import config as C
from ..data.gs_features import (
    Standardizer,
    build_feature_frame,
    feature_matrix,
    target_matrix,
)
from ..provenance import object_sha256, runtime_manifest, sha256_file
from ..train import metrics as M
from ..train.trainer import FROZEN_SEEDS


@dataclass
class XGBoostConfig:
    model_id: str = "xgboost_descriptors"
    seed: int = FROZEN_SEEDS[0]
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    n_jobs: int = 8
    development_table: str = "outputs/phase1/development_cleaned.parquet"
    development_split: str = "outputs/phase1/splits/development.json"
    data_fingerprint_file: str = "outputs/phase1/data_fingerprint.json"
    confirmatory: bool = True
    out_dir: str = "outputs/runs/xgboost_descriptors_seed0"


def _new_model(cfg: XGBoostConfig) -> MultiOutputRegressor:
    return MultiOutputRegressor(
        XGBRegressor(
            objective="reg:squarederror",
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            random_state=cfg.seed,
            n_jobs=1,
            tree_method="hist",
        ),
        n_jobs=cfg.n_jobs,
    )


def _validate(cfg: XGBoostConfig) -> None:
    if cfg.confirmatory and cfg.seed not in FROZEN_SEEDS:
        raise ValueError(f"seed {cfg.seed} is not frozen")
    expected = (300, 6, 0.05, 0.8, 0.8)
    actual = (
        cfg.n_estimators,
        cfg.max_depth,
        cfg.learning_rate,
        cfg.subsample,
        cfg.colsample_bytree,
    )
    if cfg.confirmatory and actual != expected:
        raise ValueError(f"XGBoost settings differ from protocol: {actual}")


def train_xgboost(cfg: XGBoostConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        _validate(cfg)
        table_path = Path(cfg.development_table)
        split_path = Path(cfg.development_split)
        fingerprint_path = Path(cfg.data_fingerprint_file)
        table = pd.read_parquet(table_path)
        split = json.loads(split_path.read_text())
        if "test" in split:
            raise RuntimeError("development split contains test IDs")
        fingerprint = json.loads(fingerprint_path.read_text())
        if sha256_file(table_path) != fingerprint["development_table_sha256"]:
            raise RuntimeError("development table hash mismatch")
        if sha256_file(split_path) != fingerprint["development_split_sha256"]:
            raise RuntimeError("development split hash mismatch")
        runtime = runtime_manifest(C.REPO_ROOT)
        if cfg.confirmatory and runtime["git_dirty"]:
            raise RuntimeError("confirmatory training requires clean Git worktree")

        features = build_feature_frame(table)
        train_x, names = feature_matrix(features, split["train"])
        val_x, val_names = feature_matrix(features, split["val"])
        if names != list(C.PARENT_DESCRIPTOR_COLS) or val_names != names:
            raise RuntimeError(f"unexpected descriptor order: {names}")
        standardizer = Standardizer().fit(train_x)
        train_x = standardizer.transform(train_x)
        val_x = standardizer.transform(val_x)

        models = {}
        validation_prediction = {}
        validation_target = {}
        for solvent in C.SOLVENTS:
            train_y = target_matrix(table, split["train"], solvent)
            val_y = target_matrix(table, split["val"], solvent)
            target_median = np.nanmedian(train_y, axis=0)
            train_y_fit = np.where(np.isnan(train_y), target_median, train_y)
            model = _new_model(cfg)
            model.fit(train_x, train_y_fit)
            prediction = model.predict(val_x)
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"nonfinite {solvent} validation prediction")
            if np.any(prediction <= 0) or np.any(prediction > 15):
                raise FloatingPointError(
                    f"{solvent} validation energy outside (0, 15] eV"
                )
            models[solvent] = model
            validation_prediction[solvent] = prediction
            validation_target[solvent] = val_y

        joint_prediction = np.stack(
            [validation_prediction[solvent] for solvent in C.SOLVENTS], axis=1
        )
        joint_target = np.stack(
            [validation_target[solvent] for solvent in C.SOLVENTS], axis=1
        )
        artifact = {
            "models": models,
            "standardizer": standardizer,
            "feature_names": names,
            "cfg": asdict(cfg),
            "config_sha256": object_sha256(asdict(cfg)),
            "data_fingerprint_sha256": object_sha256(fingerprint),
        }
        model_path = out_dir / "model.joblib"
        temporary = out_dir / "model.joblib.tmp"
        joblib.dump(artifact, temporary)
        os.replace(temporary, model_path)
        np.savez_compressed(
            out_dir / "validation_energy_predictions.npz",
            molecule_ids=np.asarray(split["val"], dtype=str),
            gasphase_prediction=validation_prediction["gasphase"],
            acetone_prediction=validation_prediction["acetone"],
            gasphase_target=validation_target["gasphase"],
            acetone_target=validation_target["acetone"],
        )
        report = {
            "status": "valid",
            "scope": "training_and_validation_only",
            "config": asdict(cfg),
            "config_sha256": artifact["config_sha256"],
            "data_fingerprint_sha256": artifact["data_fingerprint_sha256"],
            "feature_names": names,
            "final_validation": {
                "joint_molecule_mae_eV": M.joint_molecule_energy_mae(
                    joint_prediction, joint_target
                ),
                "gasphase_mae_eV": M.mae(
                    validation_prediction["gasphase"],
                    validation_target["gasphase"],
                ),
                "acetone_mae_eV": M.mae(
                    validation_prediction["acetone"],
                    validation_target["acetone"],
                ),
            },
            "model_sha256": sha256_file(model_path),
            "validation_predictions_sha256": sha256_file(
                out_dir / "validation_energy_predictions.npz"
            ),
            "runtime": runtime,
            "command": sys.argv,
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            "[xgboost] validation-only complete: joint MAE="
            f"{report['final_validation']['joint_molecule_mae_eV']:.6f} eV"
        )
        return report
    except BaseException as exc:
        failure = {
            "status": "failed",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "config": asdict(cfg),
        }
        (out_dir / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
