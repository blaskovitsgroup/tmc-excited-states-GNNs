"""Retrain one published configuration in a separate reproduction folder."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS = ("xgboost_descriptors", "graph_baseline", "graph_unatq", "graph_dnatq",
          "schnet_3d", "painn_3d", "painn_baseline_fusion", "painn_unatq_fusion",
          "painn_dnatq_fusion")
SEEDS = (2737188456, 1409281222, 2696188251, 3663425383, 451851728)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = ROOT / "outputs/runs" / f"{args.model}_seed{args.seed}/run_manifest.json"
    if args.model == "xgboost_descriptors":
        source = source.with_name("report.json")
    cfg = json.loads(source.read_text())["config"].copy()
    destination = ROOT / "reproduction/runs" / f"{args.model}_seed{args.seed}"
    if destination.exists():
        parser.error(f"Reproduction run already exists: {destination}")
    phase = ROOT / "outputs/phase1"
    cfg.update(development_table=str(phase / "development_cleaned.parquet"),
               development_split=str(phase / "splits/development.json"),
               data_fingerprint_file=str(phase / "data_fingerprint.json"),
               out_dir=str(destination), confirmatory=False)
    if args.model != "xgboost_descriptors":
        cfg.update(natqg_root=str(ROOT / "source_inputs/natqg"),
                   cache_dir=str(ROOT / "reproduction/cache"), resume=False,
                   study_status=None)
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return
    os.environ["TMQMG_SOURCE_ROOT"] = str(ROOT / "source_inputs")
    os.environ["TMQMG_OUTPUT_DIR"] = str(ROOT / "reproduction")
    if args.model == "xgboost_descriptors":
        from tmqmg_es.models.baselines import XGBoostConfig, train_xgboost
        train_xgboost(XGBoostConfig(**cfg))
    else:
        from tmqmg_es.train.trainer import TrainConfig, train
        train(TrainConfig(**cfg))


if __name__ == "__main__":
    main()
