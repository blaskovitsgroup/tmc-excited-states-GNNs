"""Command-line entry points for the confirmatory tmQMg* workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import config as C
from .provenance import object_sha256, sha256_file


def cmd_audit(args) -> None:
    from .data.audit import run_audit

    run_audit(out_dir=args.out)


def cmd_split(args) -> None:
    from .data.splits import (
        SPLIT_SEED,
        build_confirmatory_split,
        write_confirmatory_split,
    )

    if args.seed != SPLIT_SEED:
        raise SystemExit(f"split seed is frozen at {SPLIT_SEED}")
    out = Path(args.out)
    cleaned_path = out / "cleaned_tmQMg_star.parquet"
    cleaned = pd.read_parquet(cleaned_path).sort_values("id", kind="stable")
    split, identity, audit = build_confirmatory_split(
        cleaned, xyz_dir=C.XYZ_DIR, seed=args.seed
    )
    split_dir = out / "splits"
    write_confirmatory_split(split, identity, audit, split_dir)

    development_ids = set(split["train"]) | set(split["val"])
    development = cleaned[cleaned["id"].isin(development_ids)].copy()
    development_path = out / "development_cleaned.parquet"
    development.to_parquet(development_path, index=False)

    source_manifest = C.REPO_ROOT / "manifests/source_data_manifest.json"
    fingerprint = {
        "source_manifest_sha256": sha256_file(source_manifest),
        "cleaned_table_sha256": sha256_file(cleaned_path),
        "development_table_sha256": sha256_file(development_path),
        "development_split_sha256": sha256_file(split_dir / "development.json"),
        "locked_test_ids_sha256": sha256_file(split_dir / "locked_test_ids.json"),
        "identity_groups_sha256": sha256_file(split_dir / "identity_groups.csv"),
        "split_audit_sha256": sha256_file(split_dir / "split_audit.json"),
        "protocol_sha256": sha256_file(C.PROTOCOL_PATH),
        "preprocessing_version": "tmqmg-preprocess-1",
    }
    fingerprint_path = out / "data_fingerprint.json"
    fingerprint_path.write_text(json.dumps(fingerprint, indent=2) + "\n")
    print(
        f"[split] frozen sizes={split['sizes']} groups="
        f"{audit['identity_summary']['n_identity_groups']}"
    )
    print(f"[split] development-only table -> {development_path}")
    print(f"[split] locked test IDs -> {split_dir / 'locked_test_ids.json'}")
    print(f"[split] data fingerprint -> {fingerprint_path}")


def cmd_stats(args) -> None:
    from .data.stats import run_stats

    run_stats(out_dir=args.out)


def cmd_prepare(args) -> None:
    from .data.dataset import TMQMGDataset

    out = Path(args.out)
    cleaned = pd.read_parquet(out / "development_cleaned.parquet")
    split = json.loads((out / "splits" / "development.json").read_text())
    fingerprint = json.loads((out / "data_fingerprint.json").read_text())
    source_fingerprint = object_sha256(fingerprint)
    natqg_dir = None
    tag = "development"
    if args.natqg_type:
        natqg_dir = Path(args.natqg_root) / f"{args.natqg_type}_graphs"
        tag = f"development_{args.natqg_type}"
    development_ids = sorted(set(split["train"]) | set(split["val"]))
    dataset = TMQMGDataset(
        args.cache_dir,
        cleaned=cleaned,
        ids=development_ids,
        cutoff=args.cutoff,
        tag=tag,
        natqg_dir=natqg_dir,
        source_fingerprint=source_fingerprint,
    )
    print(
        f"[prepare] built {len(dataset)} development graphs "
        f"(cutoff={args.cutoff}, tag={tag}) -> {args.cache_dir}"
    )


def cmd_train(args) -> None:
    import yaml

    from .train.trainer import TrainConfig, train

    values = {}
    if args.config:
        values = yaml.safe_load(Path(args.config).read_text()) or {}
    for override in args.set or []:
        key, value = override.split("=", 1)
        values[key] = yaml.safe_load(value)
    train(TrainConfig(**values))


def cmd_train_xgboost(args) -> None:
    import yaml

    from .models.baselines import XGBoostConfig, train_xgboost

    values = {}
    if args.config:
        values = yaml.safe_load(Path(args.config).read_text()) or {}
    for override in args.set or []:
        key, value = override.split("=", 1)
        values[key] = yaml.safe_load(value)
    train_xgboost(XGBoostConfig(**values))


def cmd_freeze_ensembles(args) -> None:
    from .train.ensemble_manifest import freeze_ensemble_manifest

    freeze_ensemble_manifest(Path(args.runs_root), Path(args.output))


def cmd_unlock_test(args) -> None:
    from .train.locked_eval import create_unlock_record

    create_unlock_record(
        Path(args.ensemble_manifest),
        Path(args.output),
        Path(args.data_fingerprint),
        Path(args.locked_ids),
    )


def cmd_locked_predict(args) -> None:
    from .train.locked_eval import predict_locked_model

    predict_locked_model(
        args.model_id,
        Path(args.ensemble_manifest),
        Path(args.unlock),
        Path(args.data_fingerprint),
        Path(args.locked_ids),
        Path(args.cleaned_table),
        Path(args.output_dir),
    )


def cmd_locked_summarize(args) -> None:
    from .train.locked_eval import summarize_locked_predictions

    summarize_locked_predictions(
        Path(args.prediction_dir),
        Path(args.ensemble_manifest),
        Path(args.unlock),
        Path(args.output_dir),
    )


def cmd_audit_selected_outputs(args) -> None:
    from .train.selected_output_audit import audit_selected_model_outputs

    audit_selected_model_outputs(
        args.model_id,
        Path(args.runs_root),
        Path(args.development_table),
        Path(args.development_split),
        Path(args.data_fingerprint),
        Path(args.output_dir),
    )
















def cmd_all(args) -> None:
    cmd_audit(args)
    cmd_split(args)
    cmd_stats(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tmqmg-es", description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=C.PHASE1_DIR,
        help="data-audit and split artifact directory",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "audit", help="build cleaned table and integrity records"
    ).set_defaults(func=cmd_audit)

    split = commands.add_parser(
        "split", help="build the one frozen identity-grouped split"
    )
    split.add_argument("--seed", type=int, default=20260724)
    split.set_defaults(func=cmd_split)

    commands.add_parser(
        "stats", help="write descriptive dataset statistics"
    ).set_defaults(func=cmd_stats)

    prepare = commands.add_parser(
        "prepare", help="build a development-only graph cache"
    )
    prepare.add_argument("--cache-dir", required=True)
    prepare.add_argument("--cutoff", type=float, default=5.0)
    prepare.add_argument("--natqg-type", default=None)
    prepare.add_argument("--natqg-root", default="source_inputs/natqg")
    prepare.set_defaults(func=cmd_prepare)

    train = commands.add_parser(
        "train", help="train and validate without loading the locked test set"
    )
    train.add_argument("--config")
    train.add_argument("--set", nargs="*", help="config overrides: key=value")
    train.set_defaults(func=cmd_train)

    xgboost = commands.add_parser(
        "train-xgboost",
        help="train and validate the descriptor baseline without test access",
    )
    xgboost.add_argument("--config")
    xgboost.add_argument("--set", nargs="*", help="config overrides: key=value")
    xgboost.set_defaults(func=cmd_train_xgboost)

    freeze = commands.add_parser(
        "freeze-ensembles",
        help="freeze valid seed membership using validation information only",
    )
    freeze.add_argument("--runs-root", default="outputs/runs")
    freeze.add_argument("--output", default="manifests/final_ensemble_manifest.json")
    freeze.set_defaults(func=cmd_freeze_ensembles)

    unlock = commands.add_parser(
        "unlock-test", help="create the one-time test unlock record"
    )
    unlock.add_argument(
        "--ensemble-manifest",
        default="manifests/final_ensemble_manifest.json",
    )
    unlock.add_argument(
        "--data-fingerprint", default="outputs/phase1/data_fingerprint.json"
    )
    unlock.add_argument(
        "--locked-ids", default="outputs/phase1/splits/locked_test_ids.json"
    )
    unlock.add_argument("--output", default="protocol/TEST_UNLOCKED.json")
    unlock.set_defaults(func=cmd_unlock_test)

    locked_predict = commands.add_parser(
        "locked-predict",
        help="generate predictions for one frozen model without computing metrics",
    )
    locked_predict.add_argument("model_id")
    locked_predict.add_argument(
        "--ensemble-manifest",
        default="manifests/final_ensemble_manifest.json",
    )
    locked_predict.add_argument("--unlock", default="protocol/TEST_UNLOCKED.json")
    locked_predict.add_argument(
        "--data-fingerprint", default="outputs/phase1/data_fingerprint.json"
    )
    locked_predict.add_argument(
        "--locked-ids", default="outputs/phase1/splits/locked_test_ids.json"
    )
    locked_predict.add_argument(
        "--cleaned-table", default="outputs/phase1/cleaned_tmQMg_star.parquet"
    )
    locked_predict.add_argument("--output-dir", default="outputs/locked_predictions")
    locked_predict.set_defaults(func=cmd_locked_predict)

    summarize = commands.add_parser(
        "locked-summarize",
        help="compute the preregistered metrics once after all predictions exist",
    )
    summarize.add_argument("--prediction-dir", default="outputs/locked_predictions")
    summarize.add_argument(
        "--ensemble-manifest",
        default="manifests/final_ensemble_manifest.json",
    )
    summarize.add_argument("--unlock", default="protocol/TEST_UNLOCKED.json")
    summarize.add_argument("--output-dir", default="outputs/results")
    summarize.set_defaults(func=cmd_locked_summarize)

    selected_outputs = commands.add_parser(
        "audit-selected-outputs",
        help="audit every validation output head in selected neural checkpoints",
    )
    selected_outputs.add_argument("model_id")
    selected_outputs.add_argument("--runs-root", default="outputs/runs")
    selected_outputs.add_argument(
        "--development-table",
        default="outputs/phase1/development_cleaned.parquet",
    )
    selected_outputs.add_argument(
        "--development-split",
        default="outputs/phase1/splits/development.json",
    )
    selected_outputs.add_argument(
        "--data-fingerprint",
        default="outputs/phase1/data_fingerprint.json",
    )
    selected_outputs.add_argument(
        "--output-dir",
        default="outputs/development_audit/selected_checkpoint_outputs",
    )
    selected_outputs.set_defaults(func=cmd_audit_selected_outputs)

    all_steps = commands.add_parser("all", help="audit + frozen split + stats")
    all_steps.add_argument("--seed", type=int, default=20260724)
    all_steps.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
