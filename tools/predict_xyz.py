"""Run the released PaiNN ensemble on a user-supplied XYZ file."""
import argparse
import contextlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predictor_webapp.app import Predictor, parse_xyz_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xyz", type=Path)
    parser.add_argument("--charge", type=int, required=True, choices=(-1, 0, 1))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new filename.")
    symbols, z, pos, _ = parse_xyz_text(args.xyz.read_text())
    from tmqmg_es.config import TRANSITION_METALS
    if sum(s in TRANSITION_METALS for s in symbols) != 1:
        parser.error("The released model is intended for mononuclear TMCs.")
    with contextlib.redirect_stdout(sys.stderr):
        predictor = Predictor(args.device)
        result = predictor.predict(z, pos, args.charge)
    result["input"] = {"xyz": args.xyz.name, "formal_charge": args.charge}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
