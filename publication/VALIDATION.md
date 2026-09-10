# Handoff validation (10 September 2026)

The public installation was tested in a newly created, isolated Python 3.11
environment inside `GitHub/.venv`, using the pinned package dependencies,
including PyTorch 2.12.1, PyG 2.7.0, and XGBoost 3.2.0. Installation and
`pip check` succeeded on Apple Silicon macOS. This is not a fresh Linux/CUDA
or Windows validation; original cluster runtime records remain in the manifests.

- Test suite after limiting the package to the nine published configurations:
  68 passed, with no skipped tests. Removed tests concerned excluded experiments.
- PaiNN browser inference self-test: five checkpoint hashes checked, with
  maximum energy differences of 9.54e-7 eV in each environment relative to
  stored ensemble predictions. Oscillator-strength differences were below
  3.4e-8; band probabilities passed the built-in tolerance checks.
- Headless JSON prediction: successful for KUJMUX with charge zero on CPU.
- The launcher was tested from another working directory. The server delivered
  the webpage and returned 30 energies and oscillator strengths for KUJMUX in
  both environments through HTTP. Malformed XYZ input returned HTTP 400.
- Startup regression tests cover a missing environment, missing assets, an
  occupied port, unavailable CUDA, and a package path containing spaces.
- Relocation check: a separate package copy in a temporary path containing spaces
  passed the PaiNN self-test when launched from `/`, with only the PaiNN assets
  supplied. The existing dependency environment was reused for this path test.
- Webapp setup and launch scripts reside in `predictor_webapp/`; launching the
  self-test directly from that directory passed after relocation of the scripts.
- Windows and POSIX interpreter paths are tested. Native Windows execution and
  Linux/CUDA execution have not been independently retested for this distribution.
- The package contains nine model configurations. Their 45 saved model artifacts
  and prediction arrays are retained; unrelated experiment artifacts are omitted.
- All 45 published model artifacts match the source checkpoint hashes. Neural
  checkpoint configuration hashes, packaged configuration hashes, ensemble
  membership, and linked report hashes were checked separately.
- All five XGBoost artifacts load with the recorded XGBoost 3.2.0 in an isolated
  validation environment. The older XGBoost 2.1.4 in the existing local environment
  could not load them reliably; use the pinned dependency rather than that older build.
- Independent energy MAE recomputation: all nine published configurations
  reproduce the reported values to their displayed precision. PaiNN acetone
  MAE is 0.12365047 eV; XGBoost acetone MAE is 0.35222775 eV.
- Figure and photophysical-table builders execute using bundled predictions.
- The separate retraining wrapper reads actual released run configurations
  and stores independent training under `reproduction/`. Full retraining was
  not repeated to prepare this handoff.

Public file names and text records use the main-package naming, with hash
references recalculated for the packaged text. Checkpoints, numerical prediction
arrays, and data tables remain unchanged. The serialized configuration hash
inside an unchanged checkpoint is retained separately from the hash of its
normalized public configuration record; evaluation checks the appropriate hash
without changing model parameters. A regression test checks this distinction
and verifies that a modified public configuration is still rejected.

Other portability edits include bundled source-input discovery, project-root
and example lookup in the PaiNN interface, and convenience scripts. This package
does not include a virtual environment, private literature PDFs, or Git history.
Training was not repeated. The original source repository was not modified.
`README.md` is now the single usage guide; the duplicate text guide was removed.
The local environment is excluded from the source-code release and file manifest.

The delivered file manifest is independently verifiable with
`python tools/verify_bundle.py`. It is a packaging checksum record, not a claim
that every published statistical conclusion was re-audited during packaging.
