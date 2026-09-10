"""Audit the relationship between acetone NTO fractions and CT predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
PREDICTION_FILE = REPO / "outputs/locked_predictions/painn_3d.npz"
OUTPUT_FILE = (
    REPO
    / "manuscript/overleaf_publication_acetone/generated_tables/nto_ct_relationship_audit.json"
)
ACETONE_INDEX = 1
CT_LABELS = np.array(["ddT", "LLCT", "MLCT", "LMCT"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ct_from_nto_fraction(fractions: np.ndarray) -> np.ndarray:
    occupied_metal = fractions[:, 0] > 0.5
    virtual_metal = fractions[:, 1] > 0.5
    return np.select(
        [
            occupied_metal & virtual_metal,
            ~occupied_metal & ~virtual_metal,
            occupied_metal & ~virtual_metal,
            ~occupied_metal & virtual_metal,
        ],
        [0, 1, 2, 3],
    ).astype(int)


def main() -> None:
    archive = np.load(PREDICTION_FILE)
    mask = archive["ct_mask"][:, ACETONE_INDEX].astype(bool) & archive[
        "nto_mask"
    ][:, ACETONE_INDEX].astype(bool)
    reference_ct = archive["ct_target"][mask, ACETONE_INDEX].astype(int)
    reference_nto = archive["nto_target"][mask, ACETONE_INDEX, :].astype(float)
    predicted_nto = archive["nto_prediction"][
        :, mask, ACETONE_INDEX, :
    ].astype(float).mean(axis=0)
    direct_ct = archive["ct_probability"][:, mask, ACETONE_INDEX, :].astype(
        float
    ).mean(axis=0).argmax(axis=1)

    reference_ct_from_nto = ct_from_nto_fraction(reference_nto)
    predicted_ct_from_nto = ct_from_nto_fraction(predicted_nto)
    if len(reference_ct) != 3_046 or not np.array_equal(
        reference_ct, reference_ct_from_nto
    ):
        raise RuntimeError("Reference CT labels do not reproduce the tmQMg* NTO rule.")

    llct_mlct_cross_error = ((reference_ct == 1) & (direct_ct == 2)) | (
        (reference_ct == 2) & (direct_ct == 1)
    )
    occupied_near_boundary = np.abs(reference_nto[:, 0] - 0.5) <= 0.10

    target_names = ("occupied", "virtual")
    target_metrics = {}
    for index, name in enumerate(target_names):
        residual = predicted_nto[:, index] - reference_nto[:, index]
        target_metrics[name] = {
            "reference_standard_deviation": float(
                np.std(reference_nto[:, index])
            ),
            "mae": float(np.mean(np.abs(residual))),
            "r_squared": float(
                1
                - np.sum(residual**2)
                / np.sum(
                    (reference_nto[:, index] - reference_nto[:, index].mean())
                    ** 2
                )
            ),
            "fraction_within_0.10": float(np.mean(np.abs(residual) <= 0.10)),
        }

    result = {
        "scope": "Descriptive acetone test-set audit using the locked PaiNN ensemble",
        "prediction_file": str(PREDICTION_FILE.relative_to(REPO)),
        "prediction_file_sha256": sha256(PREDICTION_FILE),
        "n_complexes": int(mask.sum()),
        "ct_label_order": CT_LABELS.tolist(),
        "reference_ct_labels_reproduced_by_nto_rule": int(
            np.sum(reference_ct == reference_ct_from_nto)
        ),
        "reference_ct_labels_reproduced_fraction": float(
            np.mean(reference_ct == reference_ct_from_nto)
        ),
        "painn_nto_metrics": target_metrics,
        "direct_ct_accuracy": float(np.mean(direct_ct == reference_ct)),
        "nto_threshold_ct_accuracy": float(
            np.mean(predicted_ct_from_nto == reference_ct)
        ),
        "direct_ct_and_nto_threshold_prediction_agreement": float(
            np.mean(direct_ct == predicted_ct_from_nto)
        ),
        "llct_mlct_cross_errors": int(llct_mlct_cross_error.sum()),
        "llct_mlct_cross_errors_with_occupied_fraction_within_0.10_of_boundary": int(
            np.sum(llct_mlct_cross_error & occupied_near_boundary)
        ),
        "llct_mlct_cross_error_near_boundary_fraction": float(
            np.mean(occupied_near_boundary[llct_mlct_cross_error])
        ),
        "all_complexes_occupied_near_boundary_fraction": float(
            np.mean(occupied_near_boundary)
        ),
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
