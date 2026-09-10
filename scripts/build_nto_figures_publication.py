"""Build acetone NTO-performance figures from the verified release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from build_figures_publication import LABELS, PREDICTIONS, REPO, verify_release


OUT = REPO / "manuscript/figures_publication_acetone"
ENVIRONMENT_INDEX = 1  # [gas phase, acetone]
BOOTSTRAP_SEED = 20260818
BOOTSTRAP_REPLICATES = 2_000
REPRESENTATIVE_MODEL = "painn_3d"

NEURAL_MODELS = (
    "graph_baseline",
    "graph_unatq",
    "graph_dnatq",
    "schnet_3d",
    "painn_3d",
    "painn_baseline_fusion",
    "painn_unatq_fusion",
    "painn_dnatq_fusion",
)

OCCUPIED_COLOR = "#4C78A8"
VIRTUAL_COLOR = "#E45756"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, stem: str, *, tight: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(
            OUT / f"{stem}.{extension}",
            bbox_inches="tight" if tight else None,
            facecolor="white",
        )
    plt.close(fig)


def load_nto_data(results: dict) -> tuple[dict[str, dict], np.ndarray, np.ndarray]:
    model_data: dict[str, dict] = {}
    reference_ids: np.ndarray | None = None
    reference_target: np.ndarray | None = None

    for model_id in NEURAL_MODELS:
        archive = np.load(PREDICTIONS / f"{model_id}.npz")
        mask = archive["nto_mask"][:, ENVIRONMENT_INDEX].astype(bool)
        molecule_ids = archive["molecule_ids"].astype(str)[mask]
        target = archive["nto_target"][mask, ENVIRONMENT_INDEX, :].astype(np.float64)
        prediction = archive["nto_prediction"][:, mask, ENVIRONMENT_INDEX, :].astype(
            np.float64
        ).mean(axis=0)

        if target.shape != (3_046, 2) or prediction.shape != (3_046, 2):
            raise RuntimeError(f"Unexpected acetone NTO shape for {model_id}.")
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
            raise RuntimeError(f"Nonfinite acetone NTO value for {model_id}.")
        if np.any(prediction < 0) or np.any(prediction > 1):
            raise RuntimeError(f"Out-of-range acetone NTO prediction for {model_id}.")

        if reference_ids is None:
            reference_ids = molecule_ids
            reference_target = target
        elif not np.array_equal(molecule_ids, reference_ids) or not np.array_equal(
            target, reference_target
        ):
            raise RuntimeError("The locked NTO targets are inconsistent across models.")

        residual = prediction - target
        mae = np.mean(np.abs(residual), axis=0)
        aggregate_mae = float(np.mean(np.abs(residual)))
        expected = results["models"][model_id]["secondary"]["solvents"]["acetone"][
            "nto"
        ]["metal_fraction_mae"]
        if not np.isclose(aggregate_mae, expected, atol=1e-12):
            raise RuntimeError(f"NTO MAE does not reproduce the release for {model_id}.")

        model_data[model_id] = {
            "prediction": prediction,
            "residual": residual,
            "absolute_error": np.abs(residual),
            "mae": mae,
            "aggregate_mae": aggregate_mae,
        }

    assert reference_ids is not None and reference_target is not None
    return model_data, reference_ids, reference_target


def bootstrap_model_mae(model_data: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
    errors = np.stack(
        [model_data[model_id]["absolute_error"] for model_id in NEURAL_MODELS],
        axis=1,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty((BOOTSTRAP_REPLICATES, len(NEURAL_MODELS), 2), dtype=float)
    chunk_size = 100
    for start in range(0, BOOTSTRAP_REPLICATES, chunk_size):
        stop = min(start + chunk_size, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, errors.shape[0], size=(stop - start, errors.shape[0]))
        bootstrap[start:stop] = errors[indices].mean(axis=1)
    return np.percentile(bootstrap, 2.5, axis=0), np.percentile(bootstrap, 97.5, axis=0)


def figure_nto_model_mae(
    model_data: dict[str, dict], lower: np.ndarray, upper: np.ndarray
) -> None:
    y = np.arange(len(NEURAL_MODELS))
    offset = 0.14
    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    for index, model_id in enumerate(NEURAL_MODELS):
        occupied, virtual = model_data[model_id]["mae"]
        ax.plot(
            [occupied, virtual],
            [index - offset, index + offset],
            color="#B0B0B0",
            linewidth=1.2,
        )

        for target_index, (value, color, marker, y_value) in enumerate(
            (
                (occupied, OCCUPIED_COLOR, "o", index - offset),
                (virtual, VIRTUAL_COLOR, "s", index + offset),
            )
        ):
            error = np.array(
                [
                    [value - lower[index, target_index]],
                    [upper[index, target_index] - value],
                ]
            )
            ax.errorbar(
                value,
                y_value,
                xerr=error,
                fmt=marker,
                markersize=7.5,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.7,
                ecolor=color,
                elinewidth=1.1,
                capsize=2.5,
                zorder=3,
            )
            ax.text(
                value + 0.0014,
                y_value,
                f"{value:.3f}",
                ha="left",
                va="center",
                fontsize=9.5,
            )

    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[model_id] for model_id in NEURAL_MODELS])
    ax.invert_yaxis()
    ax.set_xlim(0.048, 0.098)
    ax.set_xlabel("NTO Metal-Fraction MAE (Acetone)")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=OCCUPIED_COLOR,
            markeredgecolor="black",
            markersize=7.5,
            label="Occupied NTO",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=VIRTUAL_COLOR,
            markeredgecolor="black",
            markersize=7.5,
            label="Virtual NTO",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.31, right=0.98, bottom=0.17, top=0.87)
    save_figure(fig, "fig07c_nto_model_mae")


def parity_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    residual_sum_of_squares = float(np.sum(residual**2))
    total_sum_of_squares = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r_squared": 1.0 - residual_sum_of_squares / total_sum_of_squares,
        "pearson_r": float(np.corrcoef(target, prediction)[0, 1]),
        "mean_signed_error": float(np.mean(residual)),
        "within_0.05": float(np.mean(np.abs(residual) <= 0.05)),
        "within_0.10": float(np.mean(np.abs(residual) <= 0.10)),
    }


def figure_nto_parity(
    target: np.ndarray,
    prediction: np.ndarray,
    target_index: int,
    model_label: str,
    stem: str,
) -> dict[str, float]:
    descriptor = "Occupied" if target_index == 0 else "Virtual"
    metrics = parity_metrics(target[:, target_index], prediction[:, target_index])

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    density = ax.hexbin(
        target[:, target_index],
        prediction[:, target_index],
        gridsize=55,
        bins="log",
        cmap="viridis",
        mincnt=1,
        linewidths=0,
    )
    limits = (0.0, 1.0)
    ax.plot(limits, limits, linestyle="--", color="white", linewidth=1.3)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal")
    ax.set_xlabel(f"TD-DFT {descriptor}-NTO Metal Fraction")
    ax.set_ylabel(f"{model_label} {descriptor}-NTO Metal Fraction")
    ax.text(
        0.04,
        0.96,
        f"MAE = {metrics['mae']:.3f}\n$R^2$ = {metrics['r_squared']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={
            "facecolor": "white",
            "edgecolor": "#666666",
            "alpha": 0.90,
            "boxstyle": "round,pad=0.25",
            "linewidth": 0.7,
        },
    )
    colorbar = fig.colorbar(density, ax=ax, fraction=0.045, pad=0.020)
    colorbar.set_label("Point Density")
    fig.subplots_adjust(left=0.19, right=0.84, bottom=0.17, top=0.98)
    save_figure(fig, stem)
    return metrics


def binned_diagnostics(
    target: np.ndarray, prediction: np.ndarray, target_index: int
) -> list[dict[str, float | int]]:
    output: list[dict[str, float | int]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + target_index + 1)
    bins = np.linspace(0.0, 1.0, 11)
    for lower, upper in zip(bins[:-1], bins[1:]):
        selected = (target[:, target_index] >= lower) & (
            target[:, target_index] < upper if upper < 1.0 else target[:, target_index] <= upper
        )
        count = int(np.sum(selected))
        if count < 20:
            continue
        reference = target[selected, target_index]
        residual = prediction[selected, target_index] - reference
        bootstrap_indices = rng.integers(
            0, count, size=(BOOTSTRAP_REPLICATES, count)
        )
        signed_bootstrap = residual[bootstrap_indices].mean(axis=1)
        absolute_bootstrap = np.abs(residual[bootstrap_indices]).mean(axis=1)
        output.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "reference_mean": float(np.mean(reference)),
                "mean_signed_error": float(np.mean(residual)),
                "mean_signed_error_ci_lower": float(np.percentile(signed_bootstrap, 2.5)),
                "mean_signed_error_ci_upper": float(np.percentile(signed_bootstrap, 97.5)),
                "mae": float(np.mean(np.abs(residual))),
                "mae_ci_lower": float(np.percentile(absolute_bootstrap, 2.5)),
                "mae_ci_upper": float(np.percentile(absolute_bootstrap, 97.5)),
            }
        )
    return output


def figure_nto_binned_diagnostics(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, list[dict[str, float | int]]]:
    diagnostics = {
        "occupied": binned_diagnostics(target, prediction, 0),
        "virtual": binned_diagnostics(target, prediction, 1),
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    for label, color, marker in (
        ("occupied", OCCUPIED_COLOR, "o"),
        ("virtual", VIRTUAL_COLOR, "s"),
    ):
        rows = diagnostics[label]
        x = np.array([row["reference_mean"] for row in rows])
        signed = np.array([row["mean_signed_error"] for row in rows])
        signed_low = np.array([row["mean_signed_error_ci_lower"] for row in rows])
        signed_high = np.array([row["mean_signed_error_ci_upper"] for row in rows])
        mae = np.array([row["mae"] for row in rows])
        mae_low = np.array([row["mae_ci_lower"] for row in rows])
        mae_high = np.array([row["mae_ci_upper"] for row in rows])

        axes[0].errorbar(
            x,
            signed,
            yerr=np.vstack([signed - signed_low, signed_high - signed]),
            color=color,
            marker=marker,
            markeredgecolor="black",
            markeredgewidth=0.6,
            linewidth=1.5,
            capsize=2.5,
            label=f"{label.capitalize()} NTO",
        )
        axes[1].errorbar(
            x,
            mae,
            yerr=np.vstack([mae - mae_low, mae_high - mae]),
            color=color,
            marker=marker,
            markeredgecolor="black",
            markeredgewidth=0.6,
            linewidth=1.5,
            capsize=2.5,
        )

    axes[0].axhline(0.0, color="#333333", linestyle="--", linewidth=1.1)
    axes[0].set_ylabel("Mean Signed Error")
    axes[1].set_ylabel("Mean Absolute Error")
    for ax in axes:
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("TD-DFT NTO Metal Fraction")
    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=2,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.17, top=0.88, wspace=0.34)
    save_figure(fig, "figS01_nto_binned_errors")
    return diagnostics


def main() -> None:
    results = verify_release()
    model_data, molecule_ids, target = load_nto_data(results)
    lower, upper = bootstrap_model_mae(model_data)
    figure_nto_model_mae(model_data, lower, upper)

    best_model = min(NEURAL_MODELS, key=lambda model_id: model_data[model_id]["aggregate_mae"])
    if best_model != "graph_unatq":
        raise RuntimeError(f"Unexpected best aggregate NTO model: {best_model}")
    prediction = model_data[REPRESENTATIVE_MODEL]["prediction"]
    occupied_metrics = figure_nto_parity(
        target,
        prediction,
        0,
        LABELS[REPRESENTATIVE_MODEL],
        "fig07a_painn_nto_occupied_parity",
    )
    virtual_metrics = figure_nto_parity(
        target,
        prediction,
        1,
        LABELS[REPRESENTATIVE_MODEL],
        "fig07b_painn_nto_virtual_parity",
    )
    binned = figure_nto_binned_diagnostics(target, prediction)

    manifest = {
        "scope": "Acetone NTO performance from the current verified locked release",
        "environment": "acetone",
        "valid_complexes": int(len(molecule_ids)),
        "target_order": ["occupied_NTO_metal_fraction", "virtual_NTO_metal_fraction"],
        "models": list(NEURAL_MODELS),
        "best_aggregate_model": best_model,
        "representative_diagnostic_model": REPRESENTATIVE_MODEL,
        "representative_selection_reason": (
            "PaiNN has NTO errors close in absolute magnitude to the lowest model "
            "errors while requiring only the XYZ geometry and formal molecular charge."
        ),
        "representative_minus_best_aggregate_mae": (
            model_data[REPRESENTATIVE_MODEL]["aggregate_mae"]
            - model_data[best_model]["aggregate_mae"]
        ),
        "bootstrap": {
            "unit": "molecule",
            "replicates": BOOTSTRAP_REPLICATES,
            "rng_seed": BOOTSTRAP_SEED,
            "interval": "percentile_95_percent",
        },
        "model_metrics": {
            model_id: {
                "occupied_mae": float(model_data[model_id]["mae"][0]),
                "virtual_mae": float(model_data[model_id]["mae"][1]),
                "aggregate_mae": model_data[model_id]["aggregate_mae"],
                "occupied_mae_ci": [float(lower[index, 0]), float(upper[index, 0])],
                "virtual_mae_ci": [float(lower[index, 1]), float(upper[index, 1])],
            }
            for index, model_id in enumerate(NEURAL_MODELS)
        },
        "representative_parity_metrics": {
            "occupied": occupied_metrics,
            "virtual": virtual_metrics,
        },
        "representative_binned_diagnostics": binned,
    }
    files = sorted(
        path
        for path in OUT.glob("*nto*")
        if path.suffix in {".pdf", ".png"}
    )
    manifest["files_sha256"] = {path.name: sha256(path) for path in files}
    (OUT / "NTO_FIGURE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Wrote verified NTO figures to {OUT}")


if __name__ == "__main__":
    main()
