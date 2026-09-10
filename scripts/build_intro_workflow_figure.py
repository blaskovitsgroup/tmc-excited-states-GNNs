"""Build draft introduction-workflow and PaiNN-architecture schematics.

The labels and layer dimensions are derived from the manuscript, model
configuration, and implementation. These figures are working assets and are
not inserted into the manuscript automatically.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "manuscript" / "introduction_figure_assets"

COLORS = {
    "navy": "#234A76",
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#4E9F4A",
    "red": "#D9534F",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "purple": "#7A5195",
    "grey": "#666666",
    "light": "#F7F7F7",
    "line": "#333333",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.linewidth": 1.0,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def box(ax, x, y, w, h, title, lines=(), color="#4C78A8", title_size=12,
        text_size=10.2, linewidth=1.2, fill_alpha=0.10, zorder=2):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=color,
            edgecolor=color,
            linewidth=linewidth,
            alpha=fill_alpha,
            zorder=zorder,
        )
    )
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor="none",
            edgecolor=color,
            linewidth=linewidth,
            zorder=zorder + 0.1,
        )
    )
    ax.text(
        x + 0.018 * w,
        y + h - 0.14 * h,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["line"],
        zorder=zorder + 1,
    )
    if lines:
        ax.text(
            x + 0.018 * w,
            y + h - 0.37 * h,
            "\n".join(lines),
            ha="left",
            va="top",
            fontsize=text_size,
            color=COLORS["line"],
            linespacing=1.28,
            zorder=zorder + 1,
        )


def arrow(ax, start, end, color=None, width=1.5, mutation=13, zorder=1):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=width,
            color=color or COLORS["grey"],
            shrinkA=2,
            shrinkB=2,
            zorder=zorder,
        )
    )


def draw_graph(ax, center, scale=1.0, directed=False, color=None):
    color = color or COLORS["teal"]
    cx, cy = center
    points = np.array(
        [
            [0.00, 0.00],
            [-0.62, 0.30],
            [0.61, 0.30],
            [-0.55, -0.42],
            [0.55, -0.42],
            [0.00, 0.69],
        ]
    )
    points *= scale
    points[:, 0] += cx
    points[:, 1] += cy
    for idx in range(1, len(points)):
        if directed:
            arrow(ax, points[0], points[idx], color=color, width=1.0,
                  mutation=8, zorder=3)
        else:
            ax.plot(
                [points[0, 0], points[idx, 0]],
                [points[0, 1], points[idx, 1]],
                color=color,
                linewidth=1.2,
                zorder=3,
            )
    for idx, (px, py) in enumerate(points):
        radius = 0.105 * scale if idx else 0.145 * scale
        face = COLORS["gold"] if idx == 0 else "white"
        ax.add_patch(
            Circle(
                (px, py),
                radius,
                facecolor=face,
                edgecolor=color,
                linewidth=1.1,
                zorder=4,
            )
        )
    ax.text(cx, cy, "M", ha="center", va="center", fontsize=8.5,
            fontweight="bold", zorder=5)


def stage_label(ax, x, text):
    ax.text(
        x,
        0.965,
        text,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=COLORS["navy"],
    )


def build_workflow():
    fig, ax = plt.subplots(figsize=(16.0, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stage_label(ax, 0.110, "Linked tmQM Data Series")
    stage_label(ax, 0.325, "Identity-Aware Data Split")
    stage_label(ax, 0.565, "Representations and Models")
    stage_label(ax, 0.865, "Excited-State Predictions")

    source_specs = [
        (0.70, "tmQM", ("CSD-derived TMC chemical space", "Mononuclear, closed shell", "Metal groups 3-12; charge -1/0/+1"), COLORS["blue"]),
        (0.44, "tmQMg", ("Optimized XYZ geometries + charge", "Nine scalar descriptors", "Connectivity, u-NatQ, d-NatQ graphs"), COLORS["green"]),
        (0.18, "tmQMg*", ("TD-DFT labels in gas phase/acetone", "30 singlet energies and intensities", "Spectral, CT, and NTO labels"), COLORS["purple"]),
    ]
    for y, title, lines, color in source_specs:
        box(ax, 0.018, y, 0.19, 0.20, title, lines, color, text_size=9.1)
    arrow(ax, (0.113, 0.70), (0.113, 0.64), color=COLORS["grey"])
    arrow(ax, (0.113, 0.44), (0.113, 0.38), color=COLORS["grey"])

    box(
        ax, 0.243, 0.59, 0.17, 0.25, "74,273 Matched TMCs",
        ("One synchronized record per complex", "XYZ, descriptors, graphs, and", "gas-phase/acetone labels linked"),
        COLORS["teal"], text_size=9.1,
    )
    arrow(ax, (0.208, 0.54), (0.243, 0.71), color=COLORS["grey"])
    box(
        ax, 0.243, 0.27, 0.17, 0.23, "Identity-Aware 80:10:10 Split",
        ("Related entries kept in one partition", "Training / validation / test", "59,417 / 7,431 / 7,425 TMCs"),
        COLORS["red"], text_size=9.0,
    )
    arrow(ax, (0.328, 0.59), (0.328, 0.50), color=COLORS["red"])

    model_x, model_w = 0.45, 0.25
    rows = [
        (0.755, "Descriptor", ("9 ground-state scalars", "XGBoost: 30 energies only"), COLORS["orange"]),
        (0.565, "Graph-Only", ("Connectivity / u-NatQ / d-NatQ", "3-layer graph MPNN"), COLORS["green"]),
        (0.375, "Coordinate-Based 3D", ("Atomic numbers + XYZ + charge", "SchNet or PaiNN"), COLORS["blue"]),
        (0.185, "Hybrid", ("PaiNN geometry + one graph branch", "Connectivity / u-NatQ / d-NatQ"), COLORS["red"]),
    ]
    branch = (0.425, 0.385)
    for y, title, lines, color in rows:
        box(ax, model_x, y, model_w, 0.145, title, lines, color,
            title_size=11.2, text_size=9.2)
        arrow(ax, branch, (model_x, y + 0.072), color=color, width=1.2)
    draw_graph(ax, branch, scale=0.030, color=COLORS["teal"])

    box(
        ax, 0.755, 0.64, 0.225, 0.25, "State-Resolved Outputs",
        (r"30 ordered excitation energies, $\Delta E_i$", r"30 oscillator strengths, $\log(1+f_i)$", "Gas-phase and acetone outputs"),
        COLORS["blue"], title_size=11.2, text_size=9.2,
    )
    box(
        ax, 0.755, 0.34, 0.225, 0.25, "Spectral and Electronic Outputs",
        ("UV/Vis/nIR presence and maxima", "Visible CT: ddT/LLCT/MLCT/LMCT", "Occupied/virtual NTO metal fractions"),
        COLORS["purple"], title_size=11.0, text_size=9.0,
    )
    box(
        ax, 0.755, 0.12, 0.225, 0.16, "Derived Analyses",
        ("Solvatochromic energy shifts", "Broadened absorption spectra"),
        COLORS["teal"], title_size=11.0, text_size=9.2,
    )
    arrow(ax, (0.70, 0.60), (0.755, 0.765), color=COLORS["blue"])
    arrow(ax, (0.70, 0.51), (0.755, 0.465), color=COLORS["purple"])
    arrow(ax, (0.70, 0.42), (0.755, 0.20), color=COLORS["teal"])

    ax.text(
        0.575, 0.075,
        "Neural models use one molecular encoder and a solvent token to generate paired gas-phase and acetone outputs.",
        ha="center", va="center", fontsize=9.2, color=COLORS["grey"],
    )
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.03, top=0.98)
    return fig


def build_painn_panel():
    fig, ax = plt.subplots(figsize=(15.0, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.975, "PaiNN Solvent-Conditioned Multitask Architecture",
        ha="center", va="top", fontsize=14, fontweight="bold", color=COLORS["navy"],
    )

    box(
        ax, 0.018, 0.59, 0.14, 0.25, "Molecular Input",
        ("Atomic numbers, Z", "XYZ coordinates, R", "Formal charge, q"),
        COLORS["blue"], title_size=11.0, text_size=9.4,
    )
    draw_graph(ax, (0.088, 0.49), scale=0.050, color=COLORS["blue"])
    ax.text(0.088, 0.405, r"Radius graph: 5.0 $\mathrm{\AA}$ cutoff", ha="center", va="top",
            fontsize=9.0, color=COLORS["grey"])

    box(
        ax, 0.19, 0.59, 0.145, 0.25, "Initial Features",
        ("Element embedding", "192 scalar channels", "192 vector channels (zero initialized)", "50 radial basis functions"),
        COLORS["teal"], title_size=11.0, text_size=8.7,
    )
    arrow(ax, (0.158, 0.715), (0.19, 0.715), color=COLORS["blue"])

    box(
        ax, 0.37, 0.53, 0.22, 0.37, "PaiNN Interaction Block x 5",
        ("1. Equivariant message passing", "   using distances and directions", "2. Scalar/vector gated update", "3. Residual feature updates", "Output: invariant scalar atom features"),
        COLORS["green"], title_size=11.0, text_size=9.0,
    )
    arrow(ax, (0.335, 0.715), (0.37, 0.715), color=COLORS["teal"])

    box(
        ax, 0.625, 0.59, 0.17, 0.25, "Metal-Aware Readout",
        ("Mean atom pooling", "Metal-atom embedding", "Metal-query attention pooling", "Concatenate: 576 features"),
        COLORS["orange"], title_size=10.8, text_size=8.8,
    )
    arrow(ax, (0.59, 0.715), (0.625, 0.715), color=COLORS["green"])

    box(
        ax, 0.83, 0.59, 0.15, 0.25, "Conditioning MLP",
        ("+ solvent embedding (192)", "+ formal charge (1)", "769 -> 256 -> 256", "SiLU + dropout"),
        COLORS["purple"], title_size=10.8, text_size=8.8,
    )
    arrow(ax, (0.795, 0.715), (0.83, 0.715), color=COLORS["orange"])

    box(
        ax, 0.79, 0.16, 0.19, 0.25, "Shared Heads",
        ("State decoder", "Band-presence head", "Regional-peak heads", "CT and NTO heads"),
        COLORS["red"], title_size=10.8, text_size=8.9,
    )
    arrow(ax, (0.905, 0.59), (0.885, 0.41), color=COLORS["purple"])

    box(
        ax, 0.22, 0.12, 0.50, 0.33,
        "Outputs for Each Solvent Token (Gas Phase or Acetone)",
        (
            r"30 ordered $\Delta E_i$ and 30 $\log(1+f_i)$",
            r"UV/Vis/nIR presence; regional $\Delta E$, $f_{\max}$, and $w$",
            "Visible CT class; occupied/virtual NTO metal fractions",
        ),
        COLORS["navy"], title_size=11.2, text_size=9.2,
    )
    arrow(ax, (0.79, 0.285), (0.72, 0.285), color=COLORS["red"])

    ax.text(
        0.50, 0.035,
        "The molecular embedding is computed once; the same prediction heads are reused with different solvent embeddings.",
        ha="center", va="bottom", fontsize=9.3, color=COLORS["grey"],
    )
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.03, top=0.98)
    return fig


def neuron_layer(ax, x, ys, color, radius=0.011, label=None, label_y=0.19):
    for y in ys:
        ax.add_patch(
            Circle(
                (x, y), radius, facecolor="white", edgecolor=color,
                linewidth=1.4, zorder=6,
            )
        )
    if label:
        ax.text(
            x, label_y, label, ha="center", va="top", fontsize=9.2,
            color=COLORS["line"], fontweight="bold",
        )


def fully_connect(ax, x1, ys1, x2, ys2, color="#777777"):
    for y1 in ys1:
        for y2 in ys2:
            ax.plot(
                [x1, x2], [y1, y2], color=color, linewidth=0.45,
                alpha=0.42, zorder=2,
            )


def build_painn_neuron_diagram():
    """PaiNN encoder followed by the implemented fully connected heads."""
    fig, ax = plt.subplots(figsize=(17.0, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.965, "PaiNN Encoder and Solvent-Conditioned Multitask MLP",
        ha="center", va="top", fontsize=14, fontweight="bold",
        color=COLORS["navy"],
    )

    # Molecular input and geometric GNN encoder.
    draw_graph(ax, (0.055, 0.59), scale=0.065, color=COLORS["blue"])
    ax.text(
        0.055, 0.445, "TMC Input", ha="center", va="top", fontsize=11,
        fontweight="bold", color=COLORS["line"],
    )
    ax.text(
        0.055, 0.395, "Atomic Numbers\nXYZ Coordinates\nFormal Charge",
        ha="center", va="top", fontsize=9.0, linespacing=1.25,
        color=COLORS["line"],
    )
    arrow(ax, (0.102, 0.59), (0.120, 0.59), color=COLORS["blue"], width=1.5)

    box(
        ax, 0.120, 0.36, 0.18, 0.44, "3D PaiNN Encoder",
        (
            r"5.0 $\mathrm{\AA}$ Radius Graph",
            "192 Scalar + 192 Vector Features",
            "Five Equivariant Interaction Blocks",
            "Invariant Atomic Features",
        ),
        COLORS["green"], title_size=11.2, text_size=9.0,
    )
    for i in range(5):
        bx = 0.143 + i * 0.028
        ax.add_patch(
            Rectangle(
                (bx, 0.425), 0.019, 0.095,
                facecolor=COLORS["green"], edgecolor=COLORS["green"],
                alpha=0.18 + 0.05 * i, linewidth=1.0, zorder=4,
            )
        )
    ax.text(
        0.210, 0.405, "Message-Passing Blocks", ha="center", va="top",
        fontsize=8.4, color=COLORS["grey"],
    )
    arrow(ax, (0.300, 0.59), (0.320, 0.59), color=COLORS["green"], width=1.5)

    box(
        ax, 0.320, 0.43, 0.135, 0.32, "Metal-Aware Readout",
        (
            "Mean Pooling",
            "Metal-Atom Embedding",
            "Metal-Query Attention",
            "576 Features",
        ),
        COLORS["orange"], title_size=10.6, text_size=8.8,
    )

    # Conditioning inputs converge on the MLP input layer.
    source_x = 0.490
    input_x = 0.585
    hidden1_x = 0.675
    hidden2_x = 0.765
    mol_ys = np.linspace(0.48, 0.68, 5)
    solvent_ys = np.linspace(0.305, 0.365, 3)
    charge_ys = [0.235]
    neuron_layer(ax, source_x, mol_ys, COLORS["orange"], radius=0.009)
    ax.text(
        source_x, 0.715, "Molecular\nRepresentation",
        ha="center", va="bottom", fontsize=8.7, fontweight="bold",
        linespacing=1.1, color=COLORS["line"],
    )
    ax.text(source_x, 0.455, "576 Features", ha="center", va="top", fontsize=8.5)
    arrow(ax, (0.455, 0.59), (source_x - 0.012, 0.59), color=COLORS["orange"], width=1.3)

    neuron_layer(ax, source_x, solvent_ys, COLORS["purple"], radius=0.009)
    ax.text(
        source_x, 0.390, "Solvent Embedding", ha="center", va="bottom",
        fontsize=8.5, fontweight="bold", color=COLORS["line"],
    )
    ax.text(source_x, 0.288, "192 Features", ha="center", va="top", fontsize=8.3)
    ax.text(
        0.425, 0.335, "Gas Phase\nor Acetone", ha="center", va="center",
        fontsize=8.5, color=COLORS["line"],
    )
    arrow(ax, (0.448, 0.335), (source_x - 0.012, 0.335), color=COLORS["purple"], width=1.2)

    neuron_layer(ax, source_x, charge_ys, COLORS["red"], radius=0.009)
    ax.text(
        0.445, 0.235, "Formal Charge", ha="right", va="center",
        fontsize=8.5, fontweight="bold", color=COLORS["line"],
    )
    arrow(ax, (0.452, 0.235), (source_x - 0.012, 0.235), color=COLORS["red"], width=1.2)

    input_ys = np.linspace(0.39, 0.67, 6)
    hidden1_ys = np.linspace(0.39, 0.67, 6)
    hidden2_ys = np.linspace(0.39, 0.67, 6)
    fully_connect(ax, source_x, list(mol_ys) + list(solvent_ys) + charge_ys,
                  input_x, input_ys, COLORS["grey"])
    neuron_layer(
        ax, input_x, input_ys, COLORS["navy"],
        label="Concatenated Input\n769 Features", label_y=0.345,
    )
    ax.text(input_x, 0.530, r"$\vdots$", ha="center", va="center", fontsize=13, color=COLORS["grey"])

    fully_connect(ax, input_x, input_ys, hidden1_x, hidden1_ys, COLORS["blue"])
    neuron_layer(
        ax, hidden1_x, hidden1_ys, COLORS["blue"],
        label="Hidden Layer 1\n256 Neurons", label_y=0.345,
    )
    ax.text(hidden1_x, 0.530, r"$\vdots$", ha="center", va="center", fontsize=13, color=COLORS["grey"])

    fully_connect(ax, hidden1_x, hidden1_ys, hidden2_x, hidden2_ys, COLORS["teal"])
    neuron_layer(
        ax, hidden2_x, hidden2_ys, COLORS["teal"],
        label="Hidden Layer 2\n256 Neurons", label_y=0.345,
    )
    ax.text(hidden2_x, 0.530, r"$\vdots$", ha="center", va="center", fontsize=13, color=COLORS["grey"])

    ax.text(
        0.675, 0.795, "Solvent-Conditioned Fully Connected MLP",
        ha="center", va="bottom", fontsize=11.0, fontweight="bold",
        color=COLORS["navy"],
    )
    ax.text(
        0.675, 0.765, "SiLU Activation; 0.10 Dropout after the First Linear Layer",
        ha="center", va="bottom", fontsize=8.6, color=COLORS["grey"],
    )

    # Actual multitask branches from the final 256-dimensional context.
    head_specs = [
        (0.79, r"Ordered $\Delta E_i$", r"256 $\rightarrow$ 256 $\rightarrow$ 30"),
        (0.675, r"$\log(1+f_i)$", r"256 $\rightarrow$ 256 $\rightarrow$ 30"),
        (0.560, "Band Presence", r"256 $\rightarrow$ 128 $\rightarrow$ 3"),
        (0.445, "Regional Peaks", r"3 $\times$ (256 $\rightarrow$ 128 $\rightarrow$ 3)"),
        (0.330, "Visible CT Class", r"256 $\rightarrow$ 128 $\rightarrow$ 4"),
        (0.215, "NTO Metal Fractions", r"256 $\rightarrow$ 128 $\rightarrow$ 2"),
    ]
    branch_x, branch_y = 0.805, 0.530
    ax.plot([hidden2_x + 0.012, branch_x], [branch_y, branch_y],
            color=COLORS["red"], linewidth=1.2, zorder=3)
    ax.add_patch(Circle((branch_x, branch_y), 0.005, facecolor=COLORS["red"],
                        edgecolor=COLORS["red"], zorder=5))
    for y, title, dims in head_specs:
        arrow(ax, (branch_x, branch_y), (0.830, y), color=COLORS["red"], width=0.9, mutation=9)
        box(
            ax, 0.830, y - 0.045, 0.155, 0.090, title, (dims,),
            COLORS["red"], title_size=9.3, text_size=8.0,
            linewidth=1.0, fill_alpha=0.07,
        )

    ax.text(
        0.908, 0.875, "Multitask Prediction Heads", ha="center", va="bottom",
        fontsize=11.0, fontweight="bold", color=COLORS["navy"],
    )
    ax.text(
        0.5, 0.075,
        "Representative neurons are shown; ellipses indicate omitted units. Every displayed neuron is fully connected to the next MLP layer.",
        ha="center", va="center", fontsize=9.0, color=COLORS["grey"],
    )

    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.03, top=0.98)
    return fig


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(OUT / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def main():
    save(build_workflow(), "intro_workflow_draft")
    save(build_painn_panel(), "painn_architecture_panel")
    save(build_painn_neuron_diagram(), "painn_gnn_mlp_neuron_diagram")
    print(f"Wrote introduction figure assets to {OUT}")


if __name__ == "__main__":
    main()
