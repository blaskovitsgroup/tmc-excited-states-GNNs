"""Audit local development-run artifacts without reading the locked test set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tmqmg_es.train.run_audit import audit_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/development_audit")
    )
    args = parser.parse_args()
    audit = audit_runs(args.runs_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "run_inventory.json"
    csv_path = args.output_dir / "run_inventory.csv"
    json_path.write_text(json.dumps(audit, indent=2) + "\n")
    columns = sorted({key for row in audit["runs"] for key in row if key != "checks"})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key != "checks"}
            for row in audit["runs"]
        )
    print(
        f"[development-audit] counts={audit['counts']} "
        f"test_information_used={audit['test_information_used']}"
    )
    print(f"[development-audit] wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
