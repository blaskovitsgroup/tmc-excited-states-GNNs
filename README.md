# Excited-state prediction for transition-metal complexes

Code, trained models, input data, and evaluation records for predicting excited-state
properties of transition-metal complexes. The study compares nine configurations:
XGBoost; Connectivity Graph, u-NatQ, d-NatQ; SchNet, PaiNN; and PaiNN combined
with each of the three graph representations.

**PaiNN requires an XYZ geometry and its formal molecular charge.** No precomputed
graph, NBO calculation, ground-state descriptors, or new TD-DFT calculation is
required. Predictions are returned for gas phase and acetone using the supplied
geometry and the five trained PaiNN ensemble members.

## Requirements

- Python 3.11 and the dependencies in `pyproject.toml`.
- macOS, Linux, or Windows with compatible Python 3.11 dependency wheels.
  Shell commands below are for macOS/Linux; Windows commands are given separately.
  Fresh installation and CPU prediction were tested on Apple Silicon macOS.
  Linux/CUDA and native Windows have not been independently retested for this package.
- The trained-model assets described below. Git source code alone does not contain weights.
- A CPU is sufficient for prediction. A compatible NVIDIA GPU or Apple Silicon GPU
  can also be used. Neural-network retraining is best run on a GPU.

The recorded cluster training environment used Linux, Python 3.11.5, an NVIDIA
L40S GPU, PyTorch 2.12.1, and PyTorch Geometric 2.7.0. The PaiNN job requested
one GPU, eight CPU cores, and 64 GB of host memory; these are training allocations,
not minimum prediction requirements. Package versions are recorded in
`requirements-cluster.txt` and `manifests/cluster_environment_freeze.txt`.
The jobs loaded CUDA 12.6, while the saved PyTorch runtime reports CUDA 13.2;
these describe the loaded module and compiled PyTorch runtime, respectively.
Cluster RDKit was 2024.09.6; the supplied fixed split was generated locally
using RDKit 2024.9.4. Do not regenerate the split with another RDKit version
and assume it is identical.

## Installation

All paths are relative to the downloaded package. No author-specific username,
home directory, cluster account, or installation location is required.

For the supplied handoff, open a terminal in the `GitHub` folder. After online
publication, users can instead download the code below, replacing `OWNER/REPOSITORY`
with the actual repository address:

```bash
git clone https://github.com/OWNER/REPOSITORY.git
cd REPOSITORY
```

Install Python 3.11 first if `python3.11 --version` is not available. Then run
these commands from the folder containing this README:

```bash
bash predictor_webapp/setup_painn.sh
bash predictor_webapp/start_painn.sh --selftest
bash predictor_webapp/start_painn.sh --device cpu --port 8765
```

All webapp setup and launch scripts are in `predictor_webapp/`. Users already
inside that directory can run `bash setup_painn.sh` followed by
`bash start_painn.sh --device cpu --port 8765` without changing directories.
On Windows, use `py -3.11 setup_painn.py` and `py -3.11 run_painn.py` there.
Keep the complete package: the app uses the shared model code, trained weights,
and manifests in the parent folder; it is not a standalone folder to copy alone.

The setup command creates `.venv` in the package root, installs the pinned
dependencies, and checks their consistency. Installation requires internet
access and may take several minutes. The launcher always uses this environment;
there is no need to activate it or to have a command named `python` on your PATH.
Run setup once, and use `bash predictor_webapp/start_painn.sh` on subsequent visits.

**Windows (PowerShell), from the package folder:**

```powershell
py -3.11 predictor_webapp/setup_painn.py
py -3.11 predictor_webapp/run_painn.py --selftest
py -3.11 predictor_webapp/run_painn.py --device cpu --port 8765
```

The Python launcher selects `.venv/Scripts/python.exe` on Windows and
`.venv/bin/python` on macOS/Linux. It can also be used on macOS/Linux with
`python3 predictor_webapp/run_painn.py`. Windows command-line JSON prediction uses:

```powershell
.venv\Scripts\python.exe tools\predict_xyz.py examples\KUJMUX.xyz --charge 0 --output predictions\KUJMUX.json
```

For the remaining commands, Windows users replace `.venv/bin/python` with
`.venv\Scripts\python.exe`.

For manual installation, the equivalent commands are:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[painn]'
.venv/bin/python -m pip check
```

Cluster-specific package builds in the environment record should not be copied
directly into a personal-computer installation. For GPU acceleration, the PyTorch
build and GPU driver must be compatible. The broader `ml` extra includes
dependencies for additional archived experiments; it is not needed for PaiNN.

### Trained-model assets

The local handoff already contains the assets. For an online Git clone, the
maintainer must supply separate download links; these have not yet been assigned.
Extract the assets into the repository root without changing their directory names.
The PaiNN application uses:

- `manifests/final_ensemble_manifest.json`: ensemble members and checkpoint hashes.
- `outputs/runs/painn_3d_seed*/best.pt`: the five selected checkpoints.
- `outputs/locked_predictions/painn_3d.npz`: stored predictions for the self-test
  and ensemble-disagreement comparison.
- `examples/ABAJOF.xyz`: the structure used by the self-test.

The full handoff also contains data and results for retraining and analysis. These
large files are excluded from ordinary Git commits. Only load model files obtained
from the trusted project distribution.

## Use PaiNN

### Local browser interface

From the repository root, start the interface:

```bash
bash predictor_webapp/start_painn.sh --device cpu --port 8765
```

Open http://127.0.0.1:8765, supply the XYZ file and formal charge, and view the results. Select gas phase or acetone to inspect the outputs. This starts a local
application, not a public web service. Some browser plotting libraries require internet access. Stop the application with Ctrl+C in the terminal.

### Command-line prediction

Predict the bundled neutral KUJMUX complex and write the results to JSON:

```bash
.venv/bin/python tools/predict_xyz.py examples/KUJMUX.xyz --charge 0 --output predictions/KUJMUX.json
```

For another complex, replace the XYZ path and supply its actual charge:

```bash
.venv/bin/python tools/predict_xyz.py /path/to/complex.xyz --charge -1 --output predictions/complex.json
```

Both environments are included in the JSON output. Choose a new output filename
for each prediction; existing results are not overwritten. Add `--device cuda`
for an available NVIDIA GPU or `--device mps` for a supported Mac.

### Input and output

Use a standard XYZ file: atom count on the first line, a comment on the second,
then one element symbol and three Cartesian coordinates in angstroms per atom.
Supply the formal charge separately: -1, 0, or +1. The model is intended for
closed-shell, mononuclear transition-metal complexes.

Use a chemically reasonable, relaxed structure, preferably from a quantum-mechanical
geometry optimization. The application does not optimize the geometry. Valid file
formatting and charge do not establish that a complex is within the training domain.

For each environment, PaiNN returns the 30 excitation energies and oscillator
strengths, regional band-presence probabilities and peak properties, visible CT
probabilities, and occupied/virtual NTO metal fractions. Regional broadness is also
an auxiliary trained output. The displayed absorption spectrum is derived from
the predicted energies and oscillator strengths, not a separate prediction head.
Excitation energies are more reliable than individual oscillator strengths;
interpret detailed intensities and spectra with the limitations reported in the
paper. Ensemble disagreement is a diagnostic, not a guarantee of accuracy. Use
the analysis scripts for the paper's spectral grids and metrics.

### Check the installation

```bash
bash predictor_webapp/start_painn.sh --selftest
```

This checks checkpoint hashes and compares the bundled example's PaiNN energies,
oscillator strengths, and band-presence outputs against the stored predictions.
To verify every file in the full handoff:

```bash
.venv/bin/python tools/verify_bundle.py
```

## Retraining and evaluation

Analyze the supplied results without retraining:

```bash
.venv/bin/python tools/check_energy_results.py
```

Inspect and then run the saved PaiNN training configuration:

```bash
.venv/bin/python tools/retrain.py painn_3d --seed 2737188456 --dry-run
.venv/bin/python tools/retrain.py painn_3d --seed 2737188456
```

New runs are written under `reproduction/`, leaving the supplied results unchanged.
Training uses the processed data and fixed partitions in `outputs/phase1/`,
and XYZ structures in `source_inputs/tmQMg-main/data/xyz/`. The 74,273 complexes
are divided into 59,417 training, 7,431 validation, and 7,425 test complexes.
Keep the supplied identity-grouped partitions for benchmark reproduction.
SMILES canonicalization does not independently validate transition-metal bonding.
Retraining on different hardware or software need not give identical results.

The five seeds are `2737188456`, `1409281222`, `2696188251`, `3663425383`, and
`451851728`. Repeat the training command with each seed to train an ensemble.
The wrapper reads the saved run configuration and changes paths and execution
bookkeeping only. It refuses to overwrite a reproduction run.

| Configuration | Training model ID |
| --- | --- |
| XGBoost | `xgboost_descriptors` |
| Connectivity Graph | `graph_baseline` |
| u-NatQ | `graph_unatq` |
| d-NatQ | `graph_dnatq` |
| SchNet | `schnet_3d` |
| PaiNN | `painn_3d` |
| PaiNN+Connectivity | `painn_baseline_fusion` |
| PaiNN+u-NatQ | `painn_unatq_fusion` |
| PaiNN+d-NatQ | `painn_dnatq_fusion` |

PaiNN and SchNet do not need the precomputed graph archives. For attributed-graph
or hybrid training, inspect and extract the archives in `source_inputs/natqg/`
into `baseline_graphs/`, `u-NatQ_graphs/`, and `d-NatQ_graphs/` there. Check archive
paths with `unzip -l` before extraction; allow tens of GB of additional disk space.
XGBoost requires the nine ground-state descriptors, not XYZ coordinates alone.

Use validation-only checkpoint selection and ensemble calibration for new runs;
the implementations are in `tmqmg_es/train/ensemble_manifest.py` and the selected
output audits. Do not reuse the original unlock records to describe a newly
trained ensemble as a new confirmatory experiment. Original cluster scripts under
`scripts/cluster/` use the submission directory or `TMQMG_REPO_ROOT` and the
environment interpreter or `TMQMG_PYTHON`. Load your cluster's required modules
before submission and supply its account/partition options. No cluster account
is needed to run the trained PaiNN model.

From the package root on a configured SLURM cluster, submit all five PaiNN seeds:

```bash
sbatch scripts/cluster/train.sbatch painn_3d
```

The resource requests are examples based on the recorded training allocation;
adapt the GPU request, memory, time limit, and scheduler options to your site.

For tests and figure generation, install the development dependencies:

```bash
.venv/bin/python -m pip install -e '.[painn,dev]'
.venv/bin/python -m pytest -q
.venv/bin/python scripts/build_figures_publication.py
.venv/bin/python scripts/build_nto_figures_publication.py
.venv/bin/python scripts/build_photophysical_tables_publication.py
```

These figure and table commands use the stored predictions, not newly trained
models. Figures go to `manuscript/figures_publication_acetone/`, and generated
tables to `manuscript/overleaf_publication_acetone/generated_tables/`. Final
manuscript layout edits are separate from numerical analysis. The package
contains only the nine model configurations listed above.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| `python: command not found` or missing `torch`/other modules | Run `bash predictor_webapp/setup_painn.sh`, then use `bash predictor_webapp/start_painn.sh`. Do not use an unrelated system or conda Python. |
| Script or file not found | The commands above start from the package root. Setup and launch scripts are inside `predictor_webapp/`; when already inside that folder, omit the `predictor_webapp/` prefix. |
| Python 3.11 not found | Install Python 3.11. A custom executable can be selected with `PYTHON=/path/to/python3.11 bash predictor_webapp/setup_painn.sh`. |
| Missing checkpoint, manifest, or prediction archive | Restore the model assets listed above. A Git source-code download without those assets is insufficient. |
| Checkpoint hash mismatch | Restore that checkpoint from the trusted distribution; do not bypass the integrity check. |
| Port 8765 already in use | Stop the previous application or run `bash predictor_webapp/start_painn.sh --port 8766`, then open http://127.0.0.1:8766. |
| Browser did not open automatically | Keep the terminal running and open the printed local URL manually. |
| GPU unavailable | Use `--device cpu`. GPU support depends on hardware, drivers, and the PyTorch build. |
| Browser charts or structure viewer missing | Check internet access for the browser's external libraries. Command-line JSON prediction does not use those libraries. |
| Output file already exists | Choose a new `--output` filename. Existing results are deliberately protected. |

The original direct command is also valid when the correct interpreter is used:

```bash
.venv/bin/python predictor_webapp/app.py --device cpu --port 8765
```

## Contents

- `tmqmg_es/`: preprocessing, graphs, models, training, and evaluation.
- `configs/`: model and training configurations.
- `protocol/`, `manifests/`: study settings, ensemble membership, and audit records.
- `outputs/runs/`: checkpoints, validation outputs, and run reports.
- `outputs/locked_predictions/`, `outputs/results/`: predictions and metrics.
- `outputs/phase1/`: processed data and fixed training/validation/test partitions.
- `source_inputs/`: source tables, XYZ files, and compressed graph archives.
- `predictor_webapp/`: local PaiNN browser interface.
- `tools/`, `examples/`: prediction, reproduction, verification, and example structures.
- `scripts/`: analysis and cluster job scripts.
- `publication/`: validation records and file-integrity manifests.
- `predictor_webapp/setup_painn.sh`, `predictor_webapp/start_painn.sh`: environment setup and reliable application launch.

## Preparing the public repository

This handoff has not been uploaded. Before publication, the maintainer must:

1. Choose a software license with the authors and retain third-party data notices.
2. Create the repository and replace the example `OWNER/REPOSITORY` address.
3. Publish the model/data assets separately and add real download links to the
   trained-model-assets section. This is required before promising clone-and-run use.
4. Test a clean clone plus those downloaded assets using setup, self-test, browser
   prediction, and command-line prediction. Never upload `.venv` or private papers.

`.gitignore` excludes large outputs, source data, manuscript files, and environments.
After reviewing the files to be published, the maintainer can run:

```bash
git init -b main
git add .
git status --short
git commit -m "Add excited-state models and reproduction workflows"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

Distribute large assets with their original relative directory paths and checksums;
do not force-add them to ordinary Git commits. `publication/BUNDLE_MANIFEST.json`
records the full handoff. `README.md` is the single installation and usage guide;
other retained records document the experiment and package validation.
#   t m c - e x c i t e d - s t a t e s - G N N s  
 