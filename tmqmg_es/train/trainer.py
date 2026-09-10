"""Leakage-resistant training and validation loop.

This module never loads locked-test IDs, builds a test loader, or writes a test
metric. Final test evaluation is implemented in ``tmqmg_es.train.locked_eval``.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .. import config as C
from ..data.dataset import TMQMGDataset
from ..models.excitonnet import TMCExcitonNet
from ..provenance import (
    object_sha256,
    runtime_manifest,
    sha256_file,
)
from . import metrics as M
from .losses import DEFAULT_WEIGHTS, MultiTaskLoss

FROZEN_SEEDS = (
    2737188456,
    1409281222,
    2696188251,
    3663425383,
    451851728,
)
VALIDATION_ENERGY_MAX_EV = 15.0
MODEL_SPECS = {
    "schnet_3d": {"backbone": "schnet", "regime": "A", "natqg_type": "d-NatQ"},
    "painn_3d": {"backbone": "painn", "regime": "A", "natqg_type": "d-NatQ"},
    "graph_baseline": {
        "backbone": "painn",
        "regime": "G",
        "natqg_type": "baseline",
    },
    "graph_unatq": {
        "backbone": "painn",
        "regime": "G",
        "natqg_type": "u-NatQ",
    },
    "graph_dnatq": {
        "backbone": "painn",
        "regime": "G",
        "natqg_type": "d-NatQ",
    },
    "painn_baseline_fusion": {
        "backbone": "painn",
        "regime": "C",
        "natqg_type": "baseline",
    },
    "painn_unatq_fusion": {
        "backbone": "painn",
        "regime": "C",
        "natqg_type": "u-NatQ",
    },
    "painn_dnatq_fusion": {
        "backbone": "painn",
        "regime": "C",
        "natqg_type": "d-NatQ",
    },
}
POST_CONFIRMATORY_RERUN_MODELS = (
    "graph_baseline",
    "graph_unatq",
    "painn_baseline_fusion",
    "painn_unatq_fusion",
)


@dataclass
class TrainConfig:
    model_id: str = "schnet_3d"
    backbone: str = "schnet"
    regime: str = "A"
    natqg_type: str = "d-NatQ"
    natqg_root: str = "source_inputs/natqg"
    natqg_layers: int = 3
    cutoff: float = 5.0
    hidden: int = 192
    n_interactions: int = 5
    cond_dim: int = 256
    batch_size: int = 256
    lr: float = 2e-4
    backbone_lr: float | None = None
    backbone_frozen_epochs: int = 0
    weight_decay: float = 1e-5
    epochs: int = 300
    patience: int = 40
    scheduler_patience: int = 10
    min_delta: float = 1e-4
    warmup_epochs: int = 5
    dropout: float = 0.1
    grad_clip: float = 5.0
    seed: int = FROZEN_SEEDS[0]
    cache_dir: str = "outputs/cache"
    num_workers: int = 4
    resume: bool = True
    use_ct: bool = True
    use_nto: bool = True
    use_peak: bool = True
    use_band: bool = True
    loss_weights: dict = field(default_factory=dict)
    checkpoint_selection: str = "energy_mae"
    checkpoint_energy_scale_eV: float = 0.15
    checkpoint_logf_scale: float = 0.05
    checkpoint_energy_fraction: float = 0.5
    development_table: str = "outputs/phase1/development_cleaned.parquet"
    development_split: str = "outputs/phase1/splits/development.json"
    data_fingerprint_file: str = "outputs/phase1/data_fingerprint.json"
    confirmatory: bool = True
    study_status: str | None = None
    require_valid_checkpoint: bool = False
    subset: int | None = None
    out_dir: str = "outputs/runs/default"


def set_reproducible_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _validate_config(cfg: TrainConfig) -> None:
    if cfg.backbone_lr is not None and cfg.backbone_lr <= 0.0:
        raise ValueError("backbone_lr must be positive when specified")
    if cfg.backbone_frozen_epochs < 0:
        raise ValueError("backbone_frozen_epochs cannot be negative")
    if cfg.backbone_frozen_epochs and cfg.backbone_lr is None:
        raise ValueError("backbone_frozen_epochs requires backbone_lr")
    if cfg.backbone_frozen_epochs >= cfg.epochs:
        raise ValueError("backbone must unfreeze before the training budget ends")
    if cfg.study_status not in (
        None,
        "post_confirmatory_rerun",
    ):
        raise ValueError(f"unknown study_status: {cfg.study_status}")
    if cfg.confirmatory and cfg.study_status is not None:
        raise ValueError("a post-confirmatory run cannot be confirmatory")
    strict_protocol = (
        cfg.confirmatory
        or cfg.study_status
        in {
            "post_confirmatory_rerun",
            }
    )
    if strict_protocol and cfg.seed not in FROZEN_SEEDS:
        raise ValueError(
            f"protocol seed {cfg.seed} is not in frozen seeds {FROZEN_SEEDS}"
        )
    if strict_protocol and cfg.subset is not None:
        raise ValueError("protocol-controlled runs cannot train on a subset")
    if cfg.checkpoint_selection not in {"energy_mae", "joint_energy_logf"}:
        raise ValueError(
            f"unknown checkpoint-selection rule: {cfg.checkpoint_selection}"
        )
    if cfg.checkpoint_energy_scale_eV <= 0 or cfg.checkpoint_logf_scale <= 0:
        raise ValueError("checkpoint normalization scales must be positive")
    if not 0 <= cfg.checkpoint_energy_fraction <= 1:
        raise ValueError("checkpoint_energy_fraction must lie in [0, 1]")
    if not strict_protocol:
        return
    if cfg.checkpoint_selection != "energy_mae":
        raise ValueError("frozen runs must select checkpoints by energy MAE")
    if cfg.backbone_lr is not None or cfg.backbone_frozen_epochs or cfg.require_valid_checkpoint:
        raise ValueError("custom optimization options are not allowed in the frozen protocol")
    if (
        cfg.study_status == "post_confirmatory_rerun"
        and cfg.model_id not in POST_CONFIRMATORY_RERUN_MODELS
    ):
        raise ValueError(f"model is outside the post-confirmatory rerun scope: {cfg.model_id}")
    if cfg.model_id not in MODEL_SPECS:
        raise ValueError(f"model_id is not in the frozen model list: {cfg.model_id}")
    actual_spec = {
        "backbone": cfg.backbone,
        "regime": cfg.regime,
        "natqg_type": cfg.natqg_type,
    }
    if actual_spec != MODEL_SPECS[cfg.model_id]:
        raise ValueError(
            f"model representation differs from protocol: {actual_spec} != "
            f"{MODEL_SPECS[cfg.model_id]}"
        )
    budget = (
        cfg.epochs,
        cfg.patience,
        cfg.scheduler_patience,
        cfg.min_delta,
        cfg.warmup_epochs,
    )
    if budget != (300, 40, 10, 1e-4, 5):
        raise ValueError("training budget differs from the frozen protocol")
    if (cfg.lr, cfg.weight_decay) != (2e-4, 1e-5):
        raise ValueError("optimizer settings differ from the frozen protocol")
    if (cfg.grad_clip, cfg.dropout, cfg.natqg_layers) != (5.0, 0.1, 3):
        raise ValueError("architecture/stability settings differ from protocol")
    if not all((cfg.use_ct, cfg.use_nto, cfg.use_peak, cfg.use_band)):
        raise ValueError("all frozen prediction heads must be enabled")
    effective_weights = {**DEFAULT_WEIGHTS, **cfg.loss_weights}
    if effective_weights != DEFAULT_WEIGHTS:
        raise ValueError(f"loss weights differ from protocol: {effective_weights}")
    expected = {
        "cutoff": 5.0,
        "hidden": 192,
        "n_interactions": 5,
        "cond_dim": 256,
        "batch_size": 256,
    }
    actual = {name: getattr(cfg, name) for name in expected}
    if actual != expected:
        raise ValueError(
            f"architecture settings differ from protocol: {actual} != {expected}"
        )


def _ct_class_weights(cleaned: pd.DataFrame, ids: list[str]) -> torch.Tensor:
    sub = cleaned.set_index("id").loc[ids]
    counts = np.array(
        [
            (sub["transition_nature_vis_gasphase"] == label).sum()
            + (sub["transition_nature_vis_acetone"] == label).sum()
            for label in C.CT_CLASSES
        ],
        dtype=float,
    )
    weights = counts.sum() / (len(C.CT_CLASSES) * np.clip(counts, 1, None))
    return torch.tensor(weights, dtype=torch.float32)


def _load_data_fingerprint(path: Path) -> tuple[str, dict]:
    payload = json.loads(path.read_text())
    required = {
        "source_manifest_sha256",
        "cleaned_table_sha256",
        "development_table_sha256",
        "development_split_sha256",
        "preprocessing_version",
    }
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"data fingerprint is missing fields: {sorted(missing)}")
    return object_sha256(payload), payload


def make_datasets(
    cleaned: pd.DataFrame,
    split: dict,
    cfg: TrainConfig,
    root: Path,
    source_fingerprint: str,
):
    """Build a development-only cache; locked-test rows are not present."""
    natqg_dir = None
    suffix = ""
    if cfg.regime in ("C", "G"):
        natqg_dir = Path(cfg.natqg_root) / f"{cfg.natqg_type}_graphs"
        suffix = f"_{cfg.natqg_type}"

    train_ids = split["train"]
    val_ids = split["val"]
    if cfg.subset:
        train_ids = train_ids[: cfg.subset]
        val_ids = val_ids[: max(1, cfg.subset // 4)]
    development_ids = sorted(set(train_ids) | set(val_ids))
    full = TMQMGDataset(
        root,
        cleaned=cleaned,
        ids=development_ids,
        cutoff=cfg.cutoff,
        tag=f"development{suffix}",
        natqg_dir=natqg_dir,
        source_fingerprint=source_fingerprint,
    )
    positions = full.id_to_index()
    missing = sorted(set(development_ids) - set(positions))
    if missing:
        raise RuntimeError(
            f"{len(missing)} development molecules are missing graph inputs; "
            f"first IDs: {missing[:10]}"
        )
    return {
        "train": full[[positions[molecule_id] for molecule_id in train_ids]],
        "val": full[[positions[molecule_id] for molecule_id in val_ids]],
    }


def build_model(cfg: TrainConfig) -> TMCExcitonNet:
    node_dim, edge_dim = 21, 16
    if cfg.regime in ("C", "G"):
        from ..data.natqg import feature_dims

        node_dim, edge_dim = feature_dims(cfg.natqg_type)
    return TMCExcitonNet(
        hidden=cfg.hidden,
        n_interactions=cfg.n_interactions,
        cutoff=cfg.cutoff,
        backbone=cfg.backbone,
        backbone_initialization_seed=cfg.seed,
        regime=cfg.regime,
        natqg_layers=cfg.natqg_layers,
        natqg_node_dim=node_dim,
        natqg_edge_dim=edge_dim,
        cond_dim=cfg.cond_dim,
        dropout=cfg.dropout,
        use_ct=cfg.use_ct,
        use_nto=cfg.use_nto,
        use_peak=cfg.use_peak,
        use_band=cfg.use_band,
    )


def _optimizer_parameter_groups(
    model: TMCExcitonNet,
    cfg: TrainConfig,
) -> tuple[list[dict], list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if cfg.backbone_lr is None:
        return (
            [{"params": trainable, "lr": cfg.lr, "group_name": "all"}],
            trainable,
            [],
        )
    if model.backbone is None:
        raise ValueError("differential backbone learning rate requires a 3D backbone")
    backbone_parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    task_parameters = [
        parameter for parameter in trainable if id(parameter) not in backbone_ids
    ]
    if not backbone_parameters or not task_parameters:
        raise RuntimeError("could not separate backbone and task parameters")
    groups = [
        {"params": task_parameters, "lr": cfg.lr, "group_name": "task"},
        {
            "params": backbone_parameters,
            "lr": 0.0 if cfg.backbone_frozen_epochs else cfg.backbone_lr,
            "group_name": "backbone",
        },
    ]
    return groups, trainable, backbone_parameters


def _set_epoch_optimization_state(
    optimizer: torch.optim.Optimizer,
    backbone_parameters: list[torch.nn.Parameter],
    cfg: TrainConfig,
    epoch: int,
) -> None:
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    if cfg.backbone_lr is None:
        if epoch < cfg.warmup_epochs:
            groups["all"]["lr"] = cfg.lr * (epoch + 1) / cfg.warmup_epochs
        return

    if epoch < cfg.warmup_epochs:
        groups["task"]["lr"] = cfg.lr * (epoch + 1) / cfg.warmup_epochs
    if epoch < cfg.backbone_frozen_epochs:
        for parameter in backbone_parameters:
            parameter.requires_grad_(False)
        groups["backbone"]["lr"] = 0.0
        return

    for parameter in backbone_parameters:
        parameter.requires_grad_(True)
    backbone_epoch = epoch - cfg.backbone_frozen_epochs
    if backbone_epoch < cfg.warmup_epochs:
        groups["backbone"]["lr"] = (
            cfg.backbone_lr * (backbone_epoch + 1) / cfg.warmup_epochs
        )


@torch.no_grad()
def evaluate_energy(model, loader, device) -> dict:
    model.eval()
    prediction = {solvent: [] for solvent in C.SOLVENTS}
    target = {solvent: [] for solvent in C.SOLVENTS}
    logf_prediction = {solvent: [] for solvent in C.SOLVENTS}
    logf_target = {solvent: [] for solvent in C.SOLVENTS}
    molecule_ids: list[str] = []
    for batch in loader:
        molecule_ids.extend(batch.mol_id)
        batch = batch.to(device)
        output = model(batch)
        for solvent_index, solvent in enumerate(C.SOLVENTS):
            prediction[solvent].append(output[solvent]["E"].cpu().numpy())
            target[solvent].append(batch.E[:, solvent_index, :].cpu().numpy())
            logf_prediction[solvent].append(
                output[solvent]["logf"].cpu().numpy()
            )
            logf_target[solvent].append(
                batch.logf[:, solvent_index, :].cpu().numpy()
            )

    pred = {key: np.concatenate(value) for key, value in prediction.items()}
    true = {key: np.concatenate(value) for key, value in target.items()}
    pred_logf = {
        key: np.concatenate(value) for key, value in logf_prediction.items()
    }
    true_logf = {key: np.concatenate(value) for key, value in logf_target.items()}
    joint_pred = np.stack([pred[solvent] for solvent in C.SOLVENTS], axis=1)
    joint_true = np.stack([true[solvent] for solvent in C.SOLVENTS], axis=1)
    joint_pred_logf = np.stack(
        [pred_logf[solvent] for solvent in C.SOLVENTS], axis=1
    )
    joint_true_logf = np.stack(
        [true_logf[solvent] for solvent in C.SOLVENTS], axis=1
    )
    return {
        "joint_molecule_mae_eV": M.joint_molecule_energy_mae(joint_pred, joint_true),
        "gasphase_mae_eV": M.mae(pred["gasphase"], true["gasphase"]),
        "acetone_mae_eV": M.mae(pred["acetone"], true["acetone"]),
        "joint_logf_mae": M.mae(joint_pred_logf, joint_true_logf),
        "gasphase_logf_mae": M.mae(
            pred_logf["gasphase"], true_logf["gasphase"]
        ),
        "acetone_logf_mae": M.mae(
            pred_logf["acetone"], true_logf["acetone"]
        ),
        "molecule_ids": molecule_ids,
        "prediction": pred,
        "target": true,
        "logf_prediction": pred_logf,
        "logf_target": true_logf,
    }


def checkpoint_score(validation: dict, cfg: TrainConfig) -> float:
    """Return the frozen validation-only checkpoint-selection score."""
    if cfg.checkpoint_selection == "energy_mae":
        return float(validation["joint_molecule_mae_eV"])
    if cfg.checkpoint_selection == "joint_energy_logf":
        energy_fraction = cfg.checkpoint_energy_fraction
        return float(
            energy_fraction
            * validation["joint_molecule_mae_eV"]
            / cfg.checkpoint_energy_scale_eV
            + (1.0 - energy_fraction)
            * validation["joint_logf_mae"]
            / cfg.checkpoint_logf_scale
        )
    raise ValueError(f"unknown checkpoint-selection rule: {cfg.checkpoint_selection}")


def _validation_prediction_is_valid(result: dict) -> tuple[bool, str | None]:
    for solvent in C.SOLVENTS:
        values = result["prediction"][solvent]
        if not np.isfinite(values).all():
            return False, f"nonfinite {solvent} validation prediction"
        if np.any(values <= 0) or np.any(values > VALIDATION_ENERGY_MAX_EV):
            return (
                False,
                f"{solvent} validation energy outside (0, "
                f"{VALIDATION_ENERGY_MAX_EV}] eV",
            )
    return True, None


def _validation_prediction_is_finite(result: dict) -> tuple[bool, str | None]:
    for solvent in C.SOLVENTS:
        if not np.isfinite(result["prediction"][solvent]).all():
            return False, f"nonfinite {solvent} validation prediction"
    return True, None


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_failure(out_dir: Path, cfg: TrainConfig, exc: BaseException) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failed",
        "failure_definition": "technical_failure_only",
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "config": asdict(cfg),
    }
    (out_dir / "failure.json").write_text(json.dumps(payload, indent=2) + "\n")


def train(cfg: TrainConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    try:
        return _train(cfg)
    except BaseException as exc:
        _write_failure(out_dir, cfg, exc)
        raise


def _train(cfg: TrainConfig) -> dict:
    _validate_config(cfg)
    set_reproducible_seed(cfg.seed)
    device = pick_device()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    development_table = Path(cfg.development_table)
    development_split = Path(cfg.development_split)
    fingerprint_file = Path(cfg.data_fingerprint_file)
    cleaned = pd.read_parquet(development_table)
    split = json.loads(development_split.read_text())
    if "test" in split:
        raise RuntimeError("development split unexpectedly contains test IDs")
    expected_ids = set(split["train"]) | set(split["val"])
    if set(cleaned["id"]) != expected_ids:
        raise RuntimeError("development table IDs do not match train/validation IDs")

    source_fingerprint, data_provenance = _load_data_fingerprint(fingerprint_file)
    if sha256_file(development_table) != data_provenance["development_table_sha256"]:
        raise RuntimeError("development table hash differs from frozen fingerprint")
    if sha256_file(development_split) != data_provenance["development_split_sha256"]:
        raise RuntimeError("development split hash differs from frozen fingerprint")

    config_payload = asdict(cfg)
    config_fingerprint = object_sha256(config_payload)
    runtime = runtime_manifest(C.REPO_ROOT)
    protocol_controlled = (
        cfg.confirmatory
        or cfg.study_status is not None
    )
    if protocol_controlled and runtime["git_dirty"]:
        raise RuntimeError("protocol-controlled training requires a clean Git worktree")
    study_status = cfg.study_status or "confirmatory"
    run_manifest = {
        "status": "running",
        "study_status": study_status,
        "config": config_payload,
        "config_sha256": config_fingerprint,
        "data_fingerprint_sha256": source_fingerprint,
        "data_provenance": data_provenance,
        "runtime": runtime,
        "command": sys.argv,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n"
    )

    datasets = make_datasets(
        cleaned,
        split,
        cfg,
        Path(cfg.cache_dir),
        source_fingerprint=source_fingerprint,
    )
    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": cfg.num_workers > 0,
    }
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(datasets["val"], batch_size=cfg.batch_size, **loader_kwargs)

    model = build_model(cfg).to(device)
    parameter_groups, trainable, backbone_parameters = _optimizer_parameter_groups(
        model, cfg
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=cfg.scheduler_patience
    )
    loss_fn = MultiTaskLoss(
        cfg.loss_weights,
        ct_class_weights=_ct_class_weights(cleaned, split["train"]).to(device),
    ).to(device)

    best_score = float("inf")
    best_epoch = -1
    best_validation_summary: dict | None = None
    epochs_without_improvement = 0
    history: list[dict] = []
    start_epoch = 0
    last_path = out_dir / "last.pt"
    if cfg.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("config_sha256") != config_fingerprint:
            raise RuntimeError("resume checkpoint/config fingerprint mismatch")
        if checkpoint.get("data_fingerprint_sha256") != source_fingerprint:
            raise RuntimeError("resume checkpoint/data fingerprint mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint.get("best_score", checkpoint["best_mae"])
        best_epoch = checkpoint["best_epoch"]
        best_validation_summary = checkpoint.get("best_validation_summary")
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        history = checkpoint["history"]
        print(
            f"[train] resumed epoch={start_epoch} "
            f"best_validation_score={best_score:.6f}@{best_epoch}"
        )

    for epoch in range(start_epoch, cfg.epochs):
        _set_epoch_optimization_state(
            optimizer,
            backbone_parameters,
            cfg,
            epoch,
        )

        model.train()
        started = time.time()
        batch_logs = []
        for batch_index, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss, logs = loss_fn(output, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"nonfinite training loss at epoch={epoch} batch={batch_index}"
                )
            loss.backward()
            for name, parameter in model.named_parameters():
                if (
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ):
                    raise FloatingPointError(
                        f"nonfinite gradient in {name} at epoch={epoch} "
                        f"batch={batch_index}"
                    )
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            batch_logs.append(logs)

        validation = evaluate_energy(model, val_loader, device)
        finite, reason = _validation_prediction_is_finite(validation)
        if not finite:
            raise FloatingPointError(reason)
        candidate_valid, candidate_failure_reason = (
            _validation_prediction_is_valid(validation)
        )
        val_mae = validation["joint_molecule_mae_eV"]
        val_score = checkpoint_score(validation, cfg)
        scheduler.step(val_score)
        component_means = {
            key: float(np.mean([row[key] for row in batch_logs]))
            for key in batch_logs[0]
        }
        history.append(
            {
                "epoch": epoch,
                "val_joint_molecule_mae_eV": val_mae,
                "val_gasphase_mae_eV": validation["gasphase_mae_eV"],
                "val_acetone_mae_eV": validation["acetone_mae_eV"],
                "val_joint_logf_mae": validation["joint_logf_mae"],
                "val_gasphase_logf_mae": validation["gasphase_logf_mae"],
                "val_acetone_logf_mae": validation["acetone_logf_mae"],
                "val_checkpoint_score": val_score,
                "train_components": component_means,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "learning_rates": {
                    group["group_name"]: group["lr"]
                    for group in optimizer.param_groups
                },
                "validation_output_valid": candidate_valid,
                "validation_output_failure_reason": candidate_failure_reason,
                "seconds": round(time.time() - started, 1),
            }
        )
        checkpoint_eligible = not cfg.require_valid_checkpoint or candidate_valid
        if checkpoint_eligible and val_score < best_score - cfg.min_delta:
            best_score = val_score
            best_epoch = epoch
            best_validation_summary = {
                "joint_molecule_mae_eV": val_mae,
                "gasphase_mae_eV": validation["gasphase_mae_eV"],
                "acetone_mae_eV": validation["acetone_mae_eV"],
                "joint_logf_mae": validation["joint_logf_mae"],
                "gasphase_logf_mae": validation["gasphase_logf_mae"],
                "acetone_logf_mae": validation["acetone_logf_mae"],
                "checkpoint_score": val_score,
            }
            epochs_without_improvement = 0
            _atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "cfg": config_payload,
                    "config_sha256": config_fingerprint,
                    "data_fingerprint_sha256": source_fingerprint,
                    "best_epoch": best_epoch,
                    "best_validation_score": best_score,
                    "best_validation_summary": best_validation_summary,
                },
                out_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        print(
            f"[e{epoch:03d}] val_joint_MAE={val_mae:.6f} eV "
            f"gas={validation['gasphase_mae_eV']:.6f} "
            f"acetone={validation['acetone_mae_eV']:.6f} "
            f"logf={validation['joint_logf_mae']:.6f} "
            f"score={val_score:.6f} valid={candidate_valid} "
            f"best={best_score:.6f}@{best_epoch}"
        )
        _atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "best_mae": best_score,
                "best_epoch": best_epoch,
                "best_validation_summary": best_validation_summary,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "cfg": config_payload,
                "config_sha256": config_fingerprint,
                "data_fingerprint_sha256": source_fingerprint,
            },
            last_path,
        )
        if epochs_without_improvement >= cfg.patience:
            print(f"[train] early stop at epoch {epoch}")
            break

    if best_epoch < 0:
        raise FloatingPointError(
            "no checkpoint satisfied the validation checkpoint-selection rule"
        )
    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    final_validation = evaluate_energy(model, val_loader, device)
    valid, reason = _validation_prediction_is_valid(final_validation)
    if not valid:
        raise FloatingPointError(reason)
    checkpoint_sha = sha256_file(out_dir / "best.pt")
    np.savez_compressed(
        out_dir / "validation_energy_predictions.npz",
        molecule_ids=np.asarray(final_validation["molecule_ids"], dtype=str),
        gasphase_prediction=final_validation["prediction"]["gasphase"],
        acetone_prediction=final_validation["prediction"]["acetone"],
        gasphase_target=final_validation["target"]["gasphase"],
        acetone_target=final_validation["target"]["acetone"],
        gasphase_logf_prediction=final_validation["logf_prediction"]["gasphase"],
        acetone_logf_prediction=final_validation["logf_prediction"]["acetone"],
        gasphase_logf_target=final_validation["logf_target"]["gasphase"],
        acetone_logf_target=final_validation["logf_target"]["acetone"],
    )
    report = {
        "status": "valid",
        "scope": "training_and_validation_only",
        "study_status": study_status,
        "config": config_payload,
        "config_sha256": config_fingerprint,
        "data_fingerprint_sha256": source_fingerprint,
        "best_epoch": best_epoch,
        "best_validation_score": checkpoint_score(final_validation, cfg),
        "best_val_joint_molecule_mae_eV": final_validation[
            "joint_molecule_mae_eV"
        ],
        "best_val_joint_logf_mae": final_validation["joint_logf_mae"],
        "checkpoint_selection": (
            "lowest_validation_joint_energy_logf_score_among_finite_range_valid_"
            "checkpoints"
            if cfg.checkpoint_selection == "joint_energy_logf"
            else (
                "lowest_validation_mae_among_finite_range_valid_checkpoints"
                if cfg.require_valid_checkpoint
                else "lowest_validation_mae_then_final_validity_check"
            )
        ),
        "checkpoint_selection_parameters": {
            "metric": cfg.checkpoint_selection,
            "energy_scale_eV": cfg.checkpoint_energy_scale_eV,
            "logf_scale": cfg.checkpoint_logf_scale,
            "energy_fraction": cfg.checkpoint_energy_fraction,
            "finite_energy_range_required": cfg.require_valid_checkpoint,
        },
        "final_validation": {
            key: value
            for key, value in final_validation.items()
            if key
            not in {
                "molecule_ids",
                "prediction",
                "target",
                "logf_prediction",
                "logf_target",
            }
        },
        "checkpoint_sha256": checkpoint_sha,
        "validation_predictions_sha256": sha256_file(
            out_dir / "validation_energy_predictions.npz"
        ),
        "history": history,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    run_manifest["status"] = "valid"
    run_manifest["checkpoint_sha256"] = checkpoint_sha
    run_manifest["best_epoch"] = best_epoch
    run_manifest["completed_runtime"] = runtime_manifest(C.REPO_ROOT)
    (out_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n"
    )
    print(
        "[train] validation-only complete: "
        f"joint energy MAE={final_validation['joint_molecule_mae_eV']:.6f} eV; "
        f"joint logf MAE={final_validation['joint_logf_mae']:.6f}; "
        "locked test was not loaded"
    )
    return report
