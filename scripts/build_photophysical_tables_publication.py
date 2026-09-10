"""Generate manuscript tables for the photophysical-target analysis.

The tables are derived only from the hash-verified current locked-test
summary. The XGBoost configuration has no photophysical
outputs beyond state energies.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "outputs/results/locked_test_results.json"
RELEASE = REPO / "manifests/locked_evaluation_release.json"
OUT = REPO / "manuscript/overleaf_publication_acetone/generated_tables"

MODELS = (
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
    "graph_baseline": "Connectivity Graph",
    "graph_unatq": "u-NatQ",
    "graph_dnatq": "d-NatQ",
    "schnet_3d": "SchNet",
    "painn_3d": "PaiNN",
    "painn_baseline_fusion": "PaiNN+Connectivity",
    "painn_unatq_fusion": "PaiNN+u-NatQ",
    "painn_dnatq_fusion": "PaiNN+d-NatQ",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_results() -> dict:
    results = json.loads(RESULTS.read_text())
    release = json.loads(RELEASE.read_text())
    expected = release["result_artifact_sha256"]["locked_test_results.json"]
    if results.get("status") != "current_release":
        raise RuntimeError("Results are not marked as the current release.")
    if sha256(RESULTS) != expected:
        raise RuntimeError("result hash does not match the release record.")
    if not set(MODELS).issubset(results["models"]):
        raise RuntimeError("A paper-facing neural model is absent from results.")
    return results


def summary_row(model: dict, solvent: str) -> list[float]:
    result = model["secondary"]["solvents"][solvent]
    return [
        result["oscillator"]["mae_logf"],
        result["oscillator"]["spearman"],
        result["spectrum"]["spectrum_cosine"],
        result["spectrum"]["spectrum_iae"],
        result["bands"]["vis"]["f1"],
        result["peaks"]["vis"]["conditional_energy_mae_eV"],
        result["ct"]["macro_f1"],
        result["nto"]["metal_fraction_mae"],
    ]


def write_main_summary(models: dict) -> None:
    rows = []
    for model_id in MODELS:
        values = summary_row(models[model_id], "acetone")
        rows.append(
            f"{LABELS[model_id]} & {values[0]:.4f} & {values[1]:.3f} & "
            f"{values[2]:.3f} & {values[3]:.3f} & {values[4]:.3f} & "
            f"{values[5]:.3f} & {values[6]:.3f} & {values[7]:.4f} \\\\"
        )
    text = r"""{\color{blue}
\begin{table}[H]
\color{blue}
\centering
\scriptsize
\setlength{\tabcolsep}{2.8pt}
\caption{{\color{blue}Acetone photophysical-target performance of the eight neural model configurations on the locked test set of 7,425 complexes. Oscillator-strength MAE is evaluated for the 30 state-resolved $\log(1+f)$ values; $\rho_f$ is their Spearman rank correlation. Cosine and IAE describe the reconstructed broadened spectra. Visible-peak $\Delta E$ MAE is evaluated for the 3,046 complexes with a defined reference visible maximum. NTO MAE is averaged over the occupied and virtual metal fractions. All values are descriptive secondary metrics.}}
\label{tab:photophysical_summary}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\toprule
Model & \makecell{$f$ MAE\\$\log(1+f)$} & $\rho_f$ & \makecell{Spectrum\\Cosine} & \makecell{Spectrum\\IAE} & \makecell{Visible\\F1\textsuperscript{\emph{a}}} & \makecell{Visible Peak\\$\Delta E$ MAE (eV)} & \makecell{CT\\Macro-F1\textsuperscript{\emph{b}}} & \makecell{NTO-Fraction\\MAE} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\vspace{2pt}

\begin{minipage}{0.96\textwidth}
\footnotesize
\raggedright
\textsuperscript{\emph{a}} $F_1=2PR/(P+R)$, where $P$ is precision, the fraction of complexes predicted to contain a qualifying visible band that truly contain one, and $R$ is recall, the fraction of complexes with a qualifying TD-DFT visible band correctly identified by the model.\\
\textsuperscript{\emph{b}} $\mathrm{Macro\text{-}F1}=\frac{1}{4}\sum_c F_{1,c}$ for $c\in\{\mathrm{LLCT},\mathrm{MLCT},\mathrm{LMCT},\mathrm{ddT}\}$, where $F_{1,c}$ is calculated from the precision and recall for CT class $c$; consequently, each CT class contributes equally irrespective of its frequency.
\end{minipage}
\end{table}
}
"""
    (OUT / "main_photophysical_summary.tex").write_text(text)


def write_band_table(models: dict) -> None:
    rows = []
    for model_id in MODELS:
        bands = models[model_id]["secondary"]["solvents"]["acetone"]["bands"]
        values = [
            bands[region][metric]
            for region in ("uv", "vis", "nir")
            for metric in ("f1", "recall")
        ]
        rows.append(
            f"{LABELS[model_id]} & "
            + " & ".join(f"{value:.3f}" for value in values)
            + r" \\"
        )
    text = r"""\begin{table*}[t!]
\centering
\small
\setlength{\tabcolsep}{5.5pt}
\renewcommand{\arraystretch}{1.12}
\caption{Acetone regional band-presence classification for the neural ensembles using a probability threshold of 0.5. The acetone test set contained 7,423 UV-positive, 3,046 visible-positive, and 52 near-IR-positive complexes among 7,425 complexes.}
\label{tab:band_metrics_acetone}
\begin{tabular}{lrrrrrr}
\toprule
& \multicolumn{2}{c}{UV} & \multicolumn{2}{c}{Visible} & \multicolumn{2}{c}{Near-IR} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Model & F1\textsuperscript{\emph{a}} & Recall & F1\textsuperscript{\emph{a}} & Recall & F1\textsuperscript{\emph{a}} & Recall \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\vspace{2pt}

\begin{minipage}{0.78\textwidth}
\footnotesize
\textsuperscript{\emph{a}} $F_1=2PR/(P+R)$, where $P$ is precision, the fraction of predicted positive cases that are correct, and $R$ is recall, the fraction of reference positive cases correctly identified by the model.
\end{minipage}
\end{table*}
"""
    (OUT / "si_band_metrics_acetone.tex").write_text(text)


def write_peak_table(models: dict) -> None:
    rows = []
    for model_id in MODELS:
        peaks = models[model_id]["secondary"]["solvents"]["acetone"]["peaks"]
        values = [
            peaks[region][metric]
            for region in ("uv", "vis", "nir")
            for metric in (
                "conditional_energy_mae_eV",
                "conditional_logf_mae",
                "conditional_logwidth_mae",
            )
        ]
        rows.append(
            f"{LABELS[model_id]} & "
            + " & ".join(f"{value:.3f}" for value in values)
            + r" \\"
        )
    text = r"""\begin{table*}[t!]
\centering
\scriptsize
\setlength{\tabcolsep}{3.0pt}
\caption{Acetone regional-maximum regression errors for the neural ensembles. Metrics are evaluated on every complex with a defined reference maximum in the indicated region, irrespective of the separate band-presence classification. The test set contained 7,423, 3,046, and 52 defined UV, visible, and near-IR maxima, respectively. $E$ is the regional-maximum energy in eV, $f$ is represented as $\log(1+f_{\max})$, and $w$ is the tmQMg* broadness descriptor represented as $\log(1+w)$.}
\label{tab:peak_metrics_acetone}
\begin{tabular}{lrrrrrrrrr}
\toprule
& \multicolumn{3}{c}{UV MAE} & \multicolumn{3}{c}{Visible MAE} & \multicolumn{3}{c}{Near-IR MAE} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Model & $E$ (eV) & $\log(1+f)$ & $\log(1+w)$ & $E$ (eV) & $\log(1+f)$ & $\log(1+w)$ & $E$ (eV) & $\log(1+f)$ & $\log(1+w)$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (OUT / "si_peak_metrics_acetone.tex").write_text(text)


def write_gas_summary(models: dict) -> None:
    rows = []
    for model_id in MODELS:
        values = summary_row(models[model_id], "gasphase")
        rows.append(
            f"{LABELS[model_id]} & {values[0]:.4f} & {values[1]:.3f} & "
            f"{values[2]:.3f} & {values[3]:.3f} & {values[4]:.3f} & "
            f"{values[5]:.3f} & {values[6]:.3f} & {values[7]:.4f} \\\\"
        )
    text = r"""\begin{table*}[t!]
\centering
\scriptsize
\setlength{\tabcolsep}{2.8pt}
\caption{Gas-phase photophysical-target performance of the eight neural model configurations. Oscillator-strength MAE is evaluated for the 30 state-resolved $\log(1+f)$ values; $\rho_f$ is their Spearman rank correlation, and cosine and IAE describe the reconstructed broadened spectra. The gas-phase test set contained 3,097 complexes with a defined visible maximum.}
\label{tab:photophysical_summary_gas}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\toprule
Model & \makecell{$f$ MAE\\$\log(1+f)$} & $\rho_f$ & \makecell{Spectrum\\Cosine} & \makecell{Spectrum\\IAE} & \makecell{Visible\\F1} & \makecell{Visible Peak\\$\Delta E$ MAE (eV)} & \makecell{CT\\Macro-F1} & \makecell{NTO-Fraction\\MAE} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\end{table*}
"""
    (OUT / "si_photophysical_summary_gas.tex").write_text(text)


def write_manifest() -> None:
    files = sorted(OUT.glob("*.tex"))
    manifest = {
        "scope": "photophysical manuscript tables",
        "source_results": str(RESULTS),
        "source_results_sha256": sha256(RESULTS),
        "release_record": str(RELEASE),
        "models_included": list(MODELS),
        "files": {path.name: sha256(path) for path in files},
    }
    (OUT / "TABLE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    results = load_verified_results()
    OUT.mkdir(parents=True, exist_ok=True)
    models = results["models"]
    write_main_summary(models)
    write_band_table(models)
    write_peak_table(models)
    write_gas_summary(models)
    write_manifest()
    print(f"Wrote verified photophysical tables to {OUT}")


if __name__ == "__main__":
    main()
