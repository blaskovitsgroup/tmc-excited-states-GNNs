"""Write a structural and covariate-balance audit for the frozen split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tmqmg_es.data.split_release_audit import build_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", type=Path, default=Path("outputs/phase1"))
    args = parser.parse_args()
    audit = build_audit(args.phase1)
    output = args.phase1 / "splits" / "split_distribution_audit.json"
    output.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"[split-release-audit] wrote {output}")


if __name__ == "__main__":
    main()
