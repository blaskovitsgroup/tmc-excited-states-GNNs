"""Build the acetone-focused manuscript figures from the release.

All numerical results are read from the verified locked-test artifacts. Dataset
composition is read from the synchronized phase-1 identifier manifest. The
underlying release remains unchanged; this script deliberately limits the
paper-facing model scope to the nine configurations selected by the author.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "outputs/results/locked_test_results.json"
RELEASE = REPO / "manifests/locked_evaluation_release.json"
LEDGER = REPO / "outputs/results/locked_evaluation_ledger.json"
IDS = REPO / "outputs/phase1/id_manifest.csv"
CLEANED = REPO / "outputs/phase1/cleaned_tmQMg_star.parquet"
PREDICTIONS = REPO / "outputs/locked_predictions"
OUT = REPO / "manuscript/figures_publication_acetone"

MODELS = (
    "xgboost_descriptors",
    "graph_baseline",
    "graph_unatq",
    "graph_dnatq",
    "schnet_3d",
    "painn_3d",
    "painn_baseline_fusion",
    "painn_unatq_fusion",
    "painn_dnatq_fusion",
)

LABELS = {
    "xgboost_descriptors": "XGBoost",
    "graph_baseline": "Connectivity Graph",
    "graph_unatq": "u-NatQ",
    "graph_dnatq": "d-NatQ",
    "schnet_3d": "SchNet",
    "painn_3d": "PaiNN",
    "painn_baseline_fusion": "PaiNN + Connectivity",
    "painn_unatq_fusion": "PaiNN + u-NatQ",
    "painn_dnatq_fusion": "PaiNN + d-NatQ",
}

COLORS = {
    "descriptor": "#D95F02",
    "graph": "#1B9E77",
    "coordinate": "#377EB8",
    "hybrid": "#E41A1C",
    "grey": "#777777",
    "light_grey": "#BDBDBD",
    "black": "#222222",
    "three_d": "#4C78A8",
    "four_d": "#F58518",
    "five_d": "#54A24B",
}

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.labelpad": 8.0,
        "axes.titlesize": 12,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "legend.frameon": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_release() -> dict:
    results = json.loads(RESULTS.read_text())
    release = json.loads(RELEASE.read_text())
    ledger = json.loads(LEDGER.read_text())
    expected = release["result_artifact_sha256"]["locked_test_results.json"]
    if results.get("status") != "current_release":
        raise RuntimeError("The results file is not marked as the current release.")
    if sha256(RESULTS) != expected:
        raise RuntimeError("The locked-test result hash does not match the release record.")
    expected_ledger = release["result_artifact_sha256"]["locked_evaluation_ledger.json"]
    if sha256(LEDGER) != expected_ledger:
        raise RuntimeError("The locked-evaluation ledger hash does not match the release record.")
    if not set(MODELS).issubset(results["models"]):
        raise RuntimeError("At least one paper-facing model is absent from the release.")
    for model_id in MODELS:
        prediction = PREDICTIONS / f"{model_id}.npz"
        if not prediction.exists():
            raise RuntimeError(f"Missing locked prediction artifact: {prediction}")
        if sha256(prediction) != ledger["prediction_sha256"][model_id]:
            raise RuntimeError(f"Locked prediction hash mismatch: {prediction}")
    return results


def model_color(model_id: str) -> str:
    if model_id == "xgboost_descriptors":
        return COLORS["descriptor"]
    if model_id.startswith("graph_"):
        return COLORS["graph"]
    if model_id.endswith("_fusion"):
        return COLORS["hybrid"]
    return COLORS["coordinate"]


def graph_representation_hatch(model_id: str) -> str:
    """Encode graph representation independently of model-family color."""
    if model_id in {"graph_unatq", "painn_unatq_fusion"}:
        return ".."
    if model_id in {"graph_dnatq", "painn_dnatq_fusion"}:
        return "///"
    return ""


def panel_label(ax: plt.Axes, label: str, x: float = 0.03, y: float = 0.97) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
    )


def save_figure(fig: plt.Figure, stem: str, *, tight: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        bbox = "tight" if tight else None
        fig.savefig(OUT / f"{stem}.{extension}", bbox_inches=bbox, facecolor="white")
    plt.close(fig)


def figure_1_dataset() -> None:
    data = pd.read_csv(IDS)
    if len(data) != 74_273 or data["id"].nunique() != 74_273:
        raise RuntimeError("The identifier manifest does not contain 74,273 unique complexes.")

    metal_order = [
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    ]
    counts = data["metal"].value_counts().reindex(metal_order)
    row_color = {
        **{metal: COLORS["three_d"] for metal in metal_order[:10]},
        **{metal: COLORS["four_d"] for metal in metal_order[10:20]},
        **{metal: COLORS["five_d"] for metal in metal_order[20:]},
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(counts))
    ax.bar(
        x,
        counts.values,
        color=[row_color[metal] for metal in counts.index],
        edgecolor="black",
        linewidth=0.7,
        width=0.78,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, rotation=45, ha="center")
    ax.set_xlabel("Metal Center")
    ax.set_ylabel("Number of Complexes")
    ax.set_xlim(-0.7, len(counts) - 0.3)
    handles = [
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=COLORS["three_d"],
               markeredgecolor="black", markersize=9, label="3d"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=COLORS["four_d"],
               markeredgecolor="black", markersize=9, label="4d"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=COLORS["five_d"],
               markeredgecolor="black", markersize=9, label="5d"),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=3)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.23, top=0.98)
    save_figure(fig, "fig01a_metal", tight=False)

    atom_counts = data["n_atoms"].astype(int)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bins = np.arange(atom_counts.min() - 0.5, atom_counts.max() + 1.5, 2)
    ax.hist(
        atom_counts,
        bins=bins,
        color=COLORS["four_d"],
        edgecolor="black",
        linewidth=0.7,
    )
    ax.set_xlabel("Number of Atoms per Complex")
    ax.set_ylabel("Number of Complexes")
    ax.set_xlim(5, 87)
    ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.18, top=0.98)
    save_figure(fig, "fig01b_size", tight=False)

    charge_counts = data["charge"].value_counts().reindex([-1, 0, 1])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(
        ["$-1$", "0", "$+1$"],
        charge_counts.values,
        color=["#4C78A8", "#54A24B", "#E45756"],
        edgecolor="black",
        linewidth=0.8,
        width=0.66,
    )
    for bar, value in zip(bars, charge_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 900, f"{value:,}",
                ha="center", va="bottom", fontsize=10.5)
    ax.set_xlabel("Formal Molecular Charge")
    ax.set_ylabel("Number of Complexes")
    ax.set_ylim(0, 69_000)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.18, top=0.98)
    save_figure(fig, "fig01c_charge", tight=False)

    ct_order = ["none", "LLCT", "MLCT", "LMCT", "ddT"]
    ct_counts = data["ct_class_acetone"].fillna("none").value_counts().reindex(ct_order)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(
        ct_order,
        ct_counts.values,
        color=["#9E9E9E", "#4C78A8", "#F58518", "#54A24B", "#E45756"],
        edgecolor="black",
        linewidth=0.8,
        width=0.70,
    )
    for bar, value in zip(bars, ct_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 800, f"{value:,}",
                ha="center", va="bottom", fontsize=9.5, rotation=0)
    ax.set_xlabel("Acetone Visible CT Class")
    ax.set_ylabel("Number of Complexes")
    ax.set_ylim(0, 49_000)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.18, top=0.98)
    save_figure(fig, "fig01d_ctclass", tight=False)


def figure_2_benchmark(results: dict) -> None:
    rows = [
        (model_id, results["models"][model_id]["energy"]["acetone"]["mae_eV"])
        for model_id in MODELS
    ]
    rows.sort(key=lambda item: item[1], reverse=True)
    y = np.arange(len(rows))

    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(7.3, 4.5),
        sharey=True,
        gridspec_kw={"width_ratios": [4.5, 1.0], "wspace": 0.05},
    )
    for ax in (left, right):
        bars = ax.barh(
            y,
            [value for _, value in rows],
            color=[model_color(model_id) for model_id, _ in rows],
            edgecolor="black",
            linewidth=0.8,
            height=0.66,
        )
        for bar, (model_id, _) in zip(bars, rows):
            bar.set_hatch(graph_representation_hatch(model_id))
        for index, (model_id, value) in enumerate(rows):
            if (value < 0.2 and ax is left) or (value > 0.2 and ax is right):
                ax.annotate(
                    f"{value:.3f}",
                    (value, index),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=10,
                )

    # Every bar begins at zero. The right panel displays only the continuation
    # of the XGBoost bar beyond the explicit axis break.
    left.set_xlim(0.0, 0.158)
    right.set_xlim(0.345, 0.358)
    left.set_yticks(y)
    left.set_yticklabels([LABELS[model_id] for model_id, _ in rows])
    left.invert_yaxis()
    left.spines["right"].set_visible(False)
    right.spines["left"].set_visible(False)
    right.tick_params(left=False)
    right.set_xticks([0.352])
    left.set_xticks([0.00, 0.05, 0.10, 0.15])
    break_style = dict(
        marker=[(-1, -0.6), (1, 0.6)],
        markersize=8,
        linestyle="none",
        color="black",
        markeredgewidth=1.0,
        clip_on=False,
    )
    left.plot([1, 1], [0, 1], transform=left.transAxes, **break_style)
    right.plot([0, 0], [0, 1], transform=right.transAxes, **break_style)
    fig.subplots_adjust(bottom=0.17)
    fig.supxlabel("$\\Delta E$ MAE (eV, Acetone)", y=0.025)

    family_handles = [
        Patch(facecolor=COLORS["descriptor"], edgecolor="black", label="Descriptor"),
        Patch(facecolor=COLORS["graph"], edgecolor="black", label="Graph Only"),
        Patch(facecolor=COLORS["coordinate"], edgecolor="black", label="3D Coordinates"),
        Patch(facecolor=COLORS["hybrid"], edgecolor="black", label="Hybrid"),
    ]
    fig.legend(
        handles=family_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=4,
    )
    save_figure(fig, "fig02_model_benchmark")


def figure_3_natq(results: dict) -> None:
    representations = [
        ("Connectivity", "graph_baseline", "painn_baseline_fusion"),
        ("u-NatQ", "graph_unatq", "painn_unatq_fusion"),
        ("d-NatQ", "graph_dnatq", "painn_dnatq_fusion"),
    ]
    acetone_mae = lambda model_id: results["models"][model_id]["energy"]["acetone"]["mae_eV"]
    painn = acetone_mae("painn_3d")
    y = np.arange(len(representations))
    fig, ax = plt.subplots(figsize=(5.8, 3.8))

    for index, (_, graph_model, hybrid_model) in enumerate(representations):
        graph_value = acetone_mae(graph_model)
        hybrid_value = acetone_mae(hybrid_model)
        ax.plot([hybrid_value, graph_value], [index, index], color="#8C8C8C", linewidth=1.4)
        ax.plot(graph_value, index, marker="o", markersize=9, linestyle="none",
                markerfacecolor=COLORS["graph"], markeredgecolor="black")
        ax.plot(hybrid_value, index, marker="o", markersize=9, linestyle="none",
                markerfacecolor=COLORS["hybrid"], markeredgecolor="black")
        ax.text(graph_value + 0.0010, index, f"{graph_value:.3f}", fontsize=10,
                ha="left", va="center")
        ax.text(hybrid_value - 0.0010, index, f"{hybrid_value:.3f}",
                fontsize=10, ha="right", va="center")

    ax.axvline(painn, color=COLORS["coordinate"], linestyle="--", linewidth=1.3)
    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _, _ in representations])
    ax.invert_yaxis()
    ax.set_xlim(0.114, 0.153)
    ax.set_xlabel("$\\Delta E$ MAE (eV, Acetone)")
    ax.set_ylabel("Graph Representation")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLORS["graph"],
               markeredgecolor="black", markersize=8, label="Graph Only"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLORS["hybrid"],
               markeredgecolor="black", markersize=8, label="PaiNN Fusion"),
        Line2D([0], [0], color=COLORS["coordinate"], linestyle="--", label="PaiNN"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    save_figure(fig, "fig03_natq_ablation")


def figure_4_state_dependence(results: dict) -> None:
    states = np.arange(1, 31)

    styles = {
        "xgboost_descriptors": (COLORS["descriptor"], "o", "-"),
        "graph_baseline": (COLORS["graph"], "o", "-"),
        "graph_unatq": (COLORS["graph"], "s", ":"),
        "graph_dnatq": (COLORS["graph"], "^", "--"),
        "schnet_3d": (COLORS["coordinate"], "^", ":"),
        "painn_3d": (COLORS["coordinate"], "D", "-"),
        "painn_baseline_fusion": (COLORS["hybrid"], "o", "-"),
    }

    state_mae = {}
    for model_id in MODELS:
        values = np.asarray(
            results["models"][model_id]["energy"]["acetone"]["mae_per_state_eV"],
            dtype=float,
        )
        expected = results["models"][model_id]["energy"]["acetone"]["mae_eV"]
        if values.shape != (30,) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"Invalid acetone per-state MAEs for {model_id}.")
        if not np.isclose(values.mean(), expected, atol=1e-12):
            raise RuntimeError(f"Per-state MAEs do not reproduce the released MAE for {model_id}.")
        state_mae[model_id] = values

    displayed_models = (
        "graph_baseline",
        "graph_unatq",
        "graph_dnatq",
        "schnet_3d",
        "painn_3d",
        "painn_baseline_fusion",
    )

    fig, ax = plt.subplots(figsize=(3.7, 4.1))
    color, marker, linestyle = styles["xgboost_descriptors"]
    ax.plot(
        states,
        state_mae["xgboost_descriptors"],
        label=LABELS["xgboost_descriptors"],
        color=color,
        marker=marker,
        markersize=3.2,
        linewidth=1.35,
        linestyle=linestyle,
        markeredgecolor="black",
        markeredgewidth=0.30,
    )
    ax.set_xlim(1, 30)
    ax.set_ylim(0.28, 0.55)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Singlet State Index")
    ax.set_ylabel("$\\Delta E$ MAE (eV, Acetone)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=1, fontsize=9.0)
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.16, top=0.82)
    save_figure(fig, "fig04a_xgboost_state_error", tight=False)

    fig, ax = plt.subplots(figsize=(3.7, 4.1))
    for model_id in displayed_models:
        color, marker, linestyle = styles[model_id]
        ax.plot(
            states,
            state_mae[model_id],
            label=LABELS[model_id],
            color=color,
            marker=marker,
            markersize=3.2,
            linewidth=1.35,
            linestyle=linestyle,
            markeredgecolor="black",
            markeredgewidth=0.30,
        )
    ax.set_xlim(1, 30)
    ax.set_ylim(0.09, 0.26)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Singlet State Index")
    ax.set_ylabel("$\\Delta E$ MAE (eV, Acetone)")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        fontsize=6.7,
        columnspacing=0.6,
        handlelength=1.6,
        handletextpad=0.4,
        labelspacing=0.35,
    )
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.16, top=0.82)
    save_figure(fig, "fig04b_neural_state_error", tight=False)


def figure_4_state_dependence_one_column(results: dict) -> None:
    """Build a one-column state-error comparison on one continuous scale."""
    states = np.arange(1, 31)
    styles = {
        "xgboost_descriptors": (COLORS["descriptor"], "o", "-"),
        "graph_baseline": (COLORS["graph"], "o", "-"),
        "graph_unatq": (COLORS["graph"], "s", ":"),
        "graph_dnatq": (COLORS["graph"], "^", "--"),
        "schnet_3d": (COLORS["coordinate"], "^", ":"),
        "painn_3d": (COLORS["coordinate"], "D", "-"),
        "painn_baseline_fusion": (COLORS["hybrid"], "o", "-"),
    }
    displayed_models = (
        "xgboost_descriptors",
        "graph_baseline",
        "graph_unatq",
        "graph_dnatq",
        "schnet_3d",
        "painn_3d",
        "painn_baseline_fusion",
    )

    fig, ax = plt.subplots(figsize=(4.0, 5.8))
    for model_id in displayed_models:
        values = np.asarray(
            results["models"][model_id]["energy"]["acetone"]["mae_per_state_eV"],
            dtype=float,
        )
        expected = results["models"][model_id]["energy"]["acetone"]["mae_eV"]
        if values.shape != (30,) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"Invalid acetone per-state MAEs for {model_id}.")
        if not np.isclose(values.mean(), expected, atol=1e-12):
            raise RuntimeError(f"Per-state MAEs do not reproduce the released MAE for {model_id}.")
        color, marker, linestyle = styles[model_id]
        ax.plot(
            states,
            values,
            label=LABELS[model_id],
            color=color,
            marker=marker,
            markersize=3.0,
            linewidth=1.35,
            linestyle=linestyle,
            markeredgecolor="black",
            markeredgewidth=0.30,
        )

    ax.set_xlim(1, 30)
    ax.set_ylim(0.09, 0.55)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Singlet State Index")
    ax.set_ylabel("$\\Delta E$ MAE (eV, Acetone)")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        fontsize=7.3,
        columnspacing=0.7,
        handlelength=1.7,
        handletextpad=0.4,
        labelspacing=0.35,
    )
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.12, top=0.82)
    save_figure(fig, "fig04_state_error_one_column", tight=False)


def figure_s_energy_distribution_by_state(results: dict) -> None:
    """Compare acetone target dispersion with PaiNN state-resolved error."""
    states = np.arange(1, 31)
    columns = [f"E_{state}_acetone_eV" for state in states]
    target = pd.read_parquet(CLEANED, columns=columns).to_numpy(dtype=float)
    if target.shape != (74_273, 30):
        raise RuntimeError("Invalid full-data acetone state-energy matrix.")
    missing_per_state = np.sum(~np.isfinite(target), axis=0)
    if missing_per_state[0] != 1 or np.any(missing_per_state[1:] != 0):
        raise RuntimeError("Unexpected missing values in the acetone state-energy matrix.")

    target_mean = np.nanmean(target, axis=0)
    target_sd = np.nanstd(target, axis=0, ddof=1)
    painn_mae = np.asarray(
        results["models"]["painn_3d"]["energy"]["acetone"]["mae_per_state_eV"],
        dtype=float,
    )
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(5.4, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.18},
    )
    top.fill_between(
        states,
        target_mean - target_sd,
        target_mean + target_sd,
        color=COLORS["coordinate"],
        alpha=0.20,
        linewidth=0,
        label="$\\pm$1 Standard Deviation",
    )
    top.plot(
        states,
        target_mean,
        color=COLORS["coordinate"],
        marker="o",
        markersize=3.0,
        linewidth=1.5,
        markeredgecolor="black",
        markeredgewidth=0.3,
        label="Mean $\\Delta E$",
    )
    top.set_ylabel("TD-DFT $\\Delta E$ (eV, Acetone)")
    top.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2, fontsize=8.5)

    bottom.plot(
        states,
        target_sd / target_sd[0],
        color=COLORS["grey"],
        marker="s",
        markersize=3.2,
        linewidth=1.5,
        markeredgecolor="black",
        markeredgewidth=0.3,
        label="TD-DFT Standard Deviation",
    )
    bottom.plot(
        states,
        painn_mae / painn_mae[0],
        color=COLORS["coordinate"],
        marker="D",
        markersize=3.0,
        linewidth=1.5,
        markeredgecolor="black",
        markeredgewidth=0.3,
        label="PaiNN MAE",
    )
    bottom.set_xlabel("Singlet State Index")
    bottom.set_ylabel("Value Relative to S$_1$")
    bottom.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2, fontsize=8.2)
    bottom.set_xlim(1, 30)
    bottom.set_xticks([1, 5, 10, 15, 20, 25, 30])
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.10, top=0.91)
    save_figure(fig, "figS_energy_distribution_by_state", tight=False)


def figure_5_parity(results: dict) -> None:
    data = np.load(PREDICTIONS / "painn_3d.npz")
    prediction = data["energy_prediction"].mean(axis=0)[:, 1, :]
    target = data["energy_target"][:, 1, :]
    valid = np.isfinite(prediction) & np.isfinite(target)
    prediction = prediction[valid]
    target = target[valid]
    mae = float(np.mean(np.abs(prediction - target)))
    expected_mae = results["models"]["painn_3d"]["energy"]["acetone"]["mae_eV"]
    if not np.isclose(mae, expected_mae, atol=1e-12):
        raise RuntimeError("Parity data do not reproduce the released acetone MAE.")
    residual_sum_of_squares = float(np.sum((prediction - target) ** 2))
    total_sum_of_squares = float(np.sum((target - np.mean(target)) ** 2))
    r_squared = 1.0 - residual_sum_of_squares / total_sum_of_squares

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    density = ax.hexbin(target, prediction, gridsize=120, bins="log", cmap="viridis",
                        mincnt=1, linewidths=0)
    limits = (0.5, 8.0)
    ax.plot(limits, limits, linestyle="--", color="white", linewidth=1.3)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal")
    ax.set_xlabel("TD-DFT $\\Delta E$ (eV, Acetone)")
    ax.set_ylabel("PaiNN $\\Delta E$ (eV, Acetone)")
    ax.text(
        0.04,
        0.96,
        f"MAE = {mae:.3f}\n$R^2$ = {r_squared:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": "#666666", "alpha": 0.90,
              "boxstyle": "round,pad=0.25", "linewidth": 0.7},
    )
    colorbar = fig.colorbar(density, ax=ax, fraction=0.045, pad=0.020)
    colorbar.set_label("Point Density")
    fig.subplots_adjust(left=0.18, right=0.90, bottom=0.16, top=0.98)
    save_figure(fig, "fig05_painn_parity", tight=False)


def figure_6_photophysical(results: dict) -> None:
    data = np.load(PREDICTIONS / "painn_3d.npz")
    logf_prediction = data["logf_prediction"].mean(axis=0)[:, 1, :]
    logf_target = data["logf_target"][:, 1, :]
    valid = np.isfinite(logf_prediction) & np.isfinite(logf_target)
    secondary = results["models"]["painn_3d"]["secondary"]["solvents"]["acetone"]

    logf_prediction_valid = logf_prediction[valid]
    logf_target_valid = logf_target[valid]
    logf_mae = float(np.mean(np.abs(logf_prediction_valid - logf_target_valid)))
    expected_logf_mae = secondary["oscillator"]["mae_logf"]
    if not np.isclose(logf_mae, expected_logf_mae, atol=1e-12):
        raise RuntimeError("Oscillator-strength parity data do not reproduce the released MAE.")
    residual_sum_of_squares = float(
        np.sum((logf_prediction_valid - logf_target_valid) ** 2)
    )
    total_sum_of_squares = float(
        np.sum((logf_target_valid - np.mean(logf_target_valid)) ** 2)
    )
    logf_r_squared = 1.0 - residual_sum_of_squares / total_sum_of_squares

    fig, oscillator_ax = plt.subplots(figsize=(4.6, 4.0))
    upper = float(np.nanpercentile(logf_target[valid], 99.9))
    density = oscillator_ax.hexbin(
        logf_target_valid,
        logf_prediction_valid,
        gridsize=120,
        bins="log",
        cmap="viridis",
        mincnt=1,
        linewidths=0,
    )
    oscillator_ax.plot([0, upper], [0, upper], linestyle="--", color="white", linewidth=1.3)
    oscillator_ax.set_xlim(0, upper)
    oscillator_ax.set_ylim(0, upper)
    oscillator_ax.set_aspect("equal")
    oscillator_ax.set_xlabel("TD-DFT $\\log(1+f)$ (Acetone)")
    oscillator_ax.set_ylabel("PaiNN $\\log(1+f)$ (Acetone)")
    oscillator_ax.text(
        0.04,
        0.96,
        f"MAE = {logf_mae:.3f}\n$R^2$ = {logf_r_squared:.3f}",
        transform=oscillator_ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": "#666666", "alpha": 0.90,
              "boxstyle": "round,pad=0.25", "linewidth": 0.7},
    )
    colorbar = fig.colorbar(density, ax=oscillator_ax, fraction=0.045, pad=0.020)
    colorbar.set_label("Point Density")
    fig.subplots_adjust(left=0.18, right=0.90, bottom=0.16, top=0.98)
    save_figure(fig, "fig06a_painn_logf_parity", tight=False)

    fig, band_ax = plt.subplots(figsize=(4.6, 4.0))
    regions = ["uv", "vis", "nir"]
    region_labels = ["UV", "Visible", "near-IR"]
    x = np.arange(3)
    width = 0.34
    auroc = [secondary["bands"][region]["auroc"] for region in regions]
    f1 = [secondary["bands"][region]["f1"] for region in regions]
    auroc_bars = band_ax.bar(
        x - width / 2,
        auroc,
        width,
        color=COLORS["coordinate"],
        edgecolor="black",
        linewidth=0.7,
        label="AUROC",
    )
    f1_bars = band_ax.bar(
        x + width / 2,
        f1,
        width,
        color=COLORS["hybrid"],
        edgecolor="black",
        linewidth=0.7,
        label="F1",
    )
    band_ax.set_xticks(x)
    band_ax.set_xticklabels(region_labels)
    band_ax.set_ylabel("Classification Score")
    band_ax.set_ylim(0, 1.08)
    fig.legend(
        handles=[auroc_bars[0], f1_bars[0]],
        labels=["AUROC", "F1"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=2,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.18, right=0.90, bottom=0.16, top=0.88)
    save_figure(fig, "fig06c_band_classification", tight=False)

    fig, ct_ax = plt.subplots(figsize=(4.6, 4.0))
    confusion = np.asarray(secondary["ct"]["confusion_matrix"], dtype=float)
    normalized = confusion / np.clip(confusion.sum(axis=1, keepdims=True), 1, None)
    image = ct_ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    classes = ["ddT", "LLCT", "MLCT", "LMCT"]
    ct_ax.set_xticks(range(4))
    ct_ax.set_xticklabels(classes, fontsize=9.5)
    ct_ax.set_yticks(range(4))
    ct_ax.set_yticklabels(classes, fontsize=9.5)
    ct_ax.set_xlabel("Predicted CT Class")
    ct_ax.set_ylabel("Reference CT Class")
    for row in range(4):
        for column in range(4):
            ct_ax.text(
                column,
                row,
                f"{normalized[row, column]:.2f}\n({int(confusion[row, column])})",
                ha="center",
                va="center",
                fontsize=8.2,
                color="white" if normalized[row, column] > 0.55 else "black",
            )
    fig.colorbar(image, ax=ct_ax, fraction=0.045, pad=0.020)
    fig.subplots_adjust(left=0.18, right=0.90, bottom=0.16, top=0.98)
    save_figure(fig, "fig06b_ct_confusion", tight=False)

    for extension in ("pdf", "png"):
        (OUT / f"fig06_photophysical_targets.{extension}").unlink(missing_ok=True)


def _gaussian_spectra(
    energies: np.ndarray, oscillator_strengths: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    energy = np.asarray(energies, dtype=float)[:, :, None]
    intensity = np.clip(np.asarray(oscillator_strengths, dtype=float), 0.0, 1e6)[
        :, :, None
    ]
    valid = np.isfinite(energy) & np.isfinite(intensity)
    energy = np.where(valid, energy, 0.0)
    intensity = np.where(valid, intensity, 0.0)
    return np.sum(
        intensity
        * np.exp(-((grid[None, None, :] - energy) ** 2) / (2 * 0.2**2)),
        axis=1,
    )


def figure_6_spectrum_examples() -> list[dict]:
    """Plot cumulative spectra for fixed-quantile PaiNN examples."""
    data = np.load(PREDICTIONS / "painn_3d.npz")
    molecule_ids = data["molecule_ids"].astype(str)
    energy_prediction = data["energy_prediction"].mean(axis=0)[:, 1, :]
    energy_target = data["energy_target"][:, 1, :]
    f_prediction = np.expm1(data["logf_prediction"].mean(axis=0)[:, 1, :])
    f_target = np.expm1(data["logf_target"][:, 1, :])

    grid = np.linspace(0.5, 8.0, 256)
    prediction_spectra = _gaussian_spectra(energy_prediction, f_prediction, grid)
    target_spectra = _gaussian_spectra(energy_target, f_target, grid)
    cosine = np.sum(prediction_spectra * target_spectra, axis=1) / (
        np.linalg.norm(prediction_spectra, axis=1)
        * np.linalg.norm(target_spectra, axis=1)
        + 1e-12
    )
    iae = np.sum(np.abs(prediction_spectra - target_spectra), axis=1) / (
        np.sum(target_spectra, axis=1) + 1e-12
    )

    visible = data["band_target"][:, 1, 1].astype(bool)
    eligible = visible & np.isfinite(cosine) & np.isfinite(iae)
    eligible_indices = np.flatnonzero(eligible)
    percentiles = (25, 50, 75)
    selection: list[dict] = []
    selected_indices: list[int] = []

    for percentile in percentiles:
        target_quantile = float(np.percentile(cosine[eligible], percentile))
        distance = np.abs(cosine[eligible] - target_quantile)
        order = np.lexsort((molecule_ids[eligible_indices], distance))
        index = int(eligible_indices[order[0]])
        selected_indices.append(index)

        selection.append(
            {
                "percentile": percentile,
                "molecule_id": molecule_ids[index],
                "target_cosine_quantile": target_quantile,
                "cosine_similarity": float(cosine[index]),
                "normalized_iae": float(iae[index]),
                "eligible_population": (
                    "acetone test complexes with a defined visible maximum"
                ),
                "eligible_count": int(np.sum(eligible)),
                "selection_rule": (
                    "nearest per-complex full-spectrum cosine percentile; "
                    "molecule ID used as deterministic tie-breaker"
                ),
            }
        )

    state_counts = (1, 10, 20, 30)
    state_labels = (
        "$S_0\\;\\rightarrow\\;S_1$",
        "$S_0\\;\\rightarrow\\;S_1$-$S_{10}$",
        "$S_0\\;\\rightarrow\\;S_1$-$S_{20}$",
        "$S_0\\;\\rightarrow\\;S_1$-$S_{30}$",
    )
    wavelength_limits = ((150, 500), (175, 550), (200, 600))
    dense_grid = np.linspace(0.5, 8.0, 2000)
    dense_wavelength = 1239.842 / dense_grid

    fig, axes = plt.subplots(4, 3, figsize=(10.2, 9.0), squeeze=False)
    for column, (index, limits) in enumerate(zip(selected_indices, wavelength_limits)):
        selection[column]["plotted_wavelength_range_nm"] = list(limits)
        selection[column]["cumulative_state_counts"] = list(state_counts)
        selection[column]["cumulative_spectrum_metrics"] = []
        for row, (state_count, state_label) in enumerate(
            zip(state_counts, state_labels)
        ):
            ax = axes[row, column]
            predicted = _gaussian_spectra(
                energy_prediction[index : index + 1, :state_count],
                f_prediction[index : index + 1, :state_count],
                dense_grid,
            )[0]
            reference = _gaussian_spectra(
                energy_target[index : index + 1, :state_count],
                f_target[index : index + 1, :state_count],
                dense_grid,
            )[0]
            predicted_metric = np.asarray(predicted, dtype=np.float64)
            reference_metric = np.asarray(reference, dtype=np.float64)
            metric_scale = max(
                float(np.max(np.abs(predicted_metric))),
                float(np.max(np.abs(reference_metric))),
                1e-300,
            )
            predicted_scaled = predicted_metric / metric_scale
            reference_scaled = reference_metric / metric_scale
            cumulative_cosine = float(
                np.sum(predicted_scaled * reference_scaled)
                / np.sqrt(
                    np.sum(predicted_scaled**2)
                    * np.sum(reference_scaled**2)
                    + 1e-24
                )
            )
            cumulative_iae = float(
                np.sum(np.abs(predicted_metric - reference_metric))
                / (np.sum(reference_metric) + 1e-12)
            )
            reference_peak_index = int(np.argmax(reference))
            prediction_peak_index = int(np.argmax(predicted))
            selection[column]["cumulative_spectrum_metrics"].append(
                {
                    "state_count": state_count,
                    "cosine_similarity": cumulative_cosine,
                    "normalized_iae": cumulative_iae,
                    "reference_peak_wavelength_nm": float(
                        dense_wavelength[reference_peak_index]
                    ),
                    "prediction_peak_wavelength_nm": float(
                        dense_wavelength[prediction_peak_index]
                    ),
                    "prediction_to_reference_peak_intensity_ratio": float(
                        predicted[prediction_peak_index]
                        / (reference[reference_peak_index] + 1e-12)
                    ),
                }
            )
            reference_scale = float(np.max(reference))
            if not np.isfinite(reference_scale) or reference_scale <= 0:
                raise RuntimeError(
                    f"Invalid reference spectrum scale for {molecule_ids[index]} "
                    f"using {state_count} states"
                )
            shown = (dense_wavelength >= limits[0]) & (dense_wavelength <= limits[1])
            order = np.argsort(dense_wavelength[shown])
            shown_wavelength = dense_wavelength[shown][order]
            reference_curve = reference[shown][order] / reference_scale
            prediction_curve = predicted[shown][order] / reference_scale

            ax.plot(
                shown_wavelength,
                reference_curve,
                color=COLORS["black"],
                linewidth=1.6,
                label="TD-DFT",
            )
            ax.plot(
                shown_wavelength,
                prediction_curve,
                color=COLORS["coordinate"],
                linewidth=1.6,
                linestyle="--",
                label="PaiNN",
            )
            ax.set_xlim(*limits)
            curve_max = max(float(np.max(reference_curve)), float(np.max(prediction_curve)))
            ax.set_ylim(0, max(1.12, 1.10 * curve_max))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            if row == 0:
                ax.set_title(
                    molecule_ids[index],
                    fontsize=10.5,
                )
            if column == 0:
                ax.text(
                    0.03,
                    0.94,
                    state_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10.5,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.88,
                        "pad": 1.5,
                    },
                )
            ax.text(
                0.97,
                0.94,
                f"Cosine = {cumulative_cosine:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.0,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.88,
                    "pad": 1.2,
                },
            )
            if row == len(state_counts) - 1:
                ax.set_xlabel("$\\lambda$ (nm)")
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if row == 0 and column == 0:
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(0.02, 0.78),
                    borderaxespad=0.2,
                    ncols=1,
                    fontsize=9.5,
                )

    fig.supylabel("Intensity/TD-DFT Maximum", x=0.012, fontsize=12)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.08, top=0.95, hspace=0.18, wspace=0.28)
    save_figure(fig, "fig06_painn_spectrum_progression", tight=False)
    return selection


def painn_raw_f_diagnostics() -> dict:
    data = np.load(PREDICTIONS / "painn_3d.npz")
    target = np.expm1(data["logf_target"][:, 1, :].astype(float))
    prediction = np.expm1(
        data["logf_prediction"].mean(axis=0)[:, 1, :].astype(float)
    )
    valid = np.isfinite(target) & np.isfinite(prediction)
    regimes = (
        ("f_lt_0.01", target < 0.01),
        ("f_0.01_to_0.05", (target >= 0.01) & (target < 0.05)),
        ("f_0.05_to_0.10", (target >= 0.05) & (target < 0.10)),
        ("f_ge_0.10", target >= 0.10),
    )
    output = {"total_values": int(np.sum(valid)), "regimes": {}}
    for name, regime in regimes:
        selected = valid & regime
        residual = prediction[selected] - target[selected]
        output["regimes"][name] = {
            "count": int(np.sum(selected)),
            "percentage": float(100 * np.sum(selected) / np.sum(valid)),
            "raw_f_mae": float(np.mean(np.abs(residual))),
            "raw_f_mean_signed_error": float(np.mean(residual)),
        }
    all_residual = prediction[valid] - target[valid]
    output["overall_raw_f_mae"] = float(np.mean(np.abs(all_residual)))
    output["overall_raw_f_mean_signed_error"] = float(np.mean(all_residual))
    return output


def figure_7_uncertainty(results: dict) -> None:
    from scipy.stats import norm

    data = np.load(PREDICTIONS / "painn_dnatq_fusion.npz")
    predictions = data["energy_prediction"]
    targets = data["energy_target"]
    mean_prediction = predictions.mean(axis=0)
    ensemble_variance = predictions.var(axis=0)
    uncertainty = results["models"]["painn_dnatq_fusion"]["uncertainty"]
    scale = uncertainty["validation_fitted_variance_scale"]
    valid = (
        np.isfinite(mean_prediction)
        & np.isfinite(targets)
        & np.isfinite(ensemble_variance)
        & (ensemble_variance > 0)
    )
    absolute_error = np.abs(mean_prediction - targets)[valid]
    raw_sd = np.sqrt(ensemble_variance[valid])
    scaled_sd = np.sqrt(scale * ensemble_variance[valid])

    levels = np.linspace(0.02, 0.995, 70)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["black"], linewidth=1.2,
            label="Ideal")
    for standard_deviation, color, label in (
        (raw_sd, COLORS["grey"], "Raw Ensemble"),
        (scaled_sd, COLORS["hybrid"], "Validation Scaled"),
    ):
        observed = [
            np.mean(absolute_error <= norm.ppf((1 + level) / 2) * standard_deviation)
            for level in levels
        ]
        ax.plot(levels, observed, color=color, linewidth=2.0, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Empirical Coverage")
    ax.legend(loc="lower right")
    save_figure(fig, "fig07_uncertainty_calibration")


def write_manifest(
    results: dict, spectrum_examples: list[dict], raw_f_diagnostics: dict
) -> None:
    files = sorted(path for path in OUT.iterdir() if path.suffix in {".pdf", ".png"})
    manifest = {
        "scope": "Acetone-focused manuscript figures",
        "source_results": str(RESULTS),
        "source_results_sha256": sha256(RESULTS),
        "release_record": str(RELEASE),
        "identifier_manifest": str(IDS),
        "identifier_manifest_sha256": sha256(IDS),
        "models_included": list(MODELS),
        "spectrum_examples": spectrum_examples,
        "painn_raw_f_diagnostics": raw_f_diagnostics,
        "files": {path.name: sha256(path) for path in files},
    }
    (OUT / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    results = verify_release()
    figure_1_dataset()
    figure_2_benchmark(results)
    figure_3_natq(results)
    figure_4_state_dependence(results)
    figure_5_parity(results)
    figure_6_photophysical(results)
    spectrum_examples = figure_6_spectrum_examples()
    raw_f_diagnostics = painn_raw_f_diagnostics()
    figure_7_uncertainty(results)
    write_manifest(results, spectrum_examples, raw_f_diagnostics)
    print(f"Wrote verified figures to {OUT}")


if __name__ == "__main__":
    main()
