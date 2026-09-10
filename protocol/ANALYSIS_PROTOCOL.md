# Frozen Confirmatory Analysis Protocol

## 1. Purpose

This protocol defines the analysis before any model is trained or evaluated
on the new test set. Its purpose is to prevent accidental test-set leakage and
post hoc choices. In simple terms, we write down the rules before seeing the
final answers, and then we follow those rules even if a different choice would
make a result look better.

is a **prospectively frozen confirmatory reevaluation of model families
selected during exploratory work**. It is not an independent discovery of
SchNet, PaiNN, or NatQ-based models. The old random test set influenced
decisions and is therefore treated as development data. No test result
is a confirmatory result.

This protocol is frozen locally because no public repository is currently
available. A local Git commit, annotated tag, and SHA-256 manifest preserve the
exact state, but they do not provide an independent third-party timestamp.
This limitation must remain stated until the unchanged history is pushed to an
external service.

## 2. Scientific task

Each input is one transition-metal complex represented by its atomic numbers,
three-dimensional coordinates, total charge, and, for specified models, a
NatQ graph. Gas-phase and acetone labels for the same complex always remain
together.

The neural models are trained to predict:

1. The first 30 ordered vertical singlet excitation energies in eV, separately
   for the gas phase and acetone.
2. The corresponding 30 oscillator strengths as `log(1 + f)`.
3. Whether a detectable UV, visible, or near-infrared band exists, using the
   tmQMg* reporting convention at `f_max >= 0.01`.
4. The energy, intensity, and reported width descriptor of each defined
   regional peak.
5. The visible-transition class (`ddT`, `LLCT`, `MLCT`, or `LMCT`) when a
   visible band is defined.
6. The occupied and virtual NTO metal fractions when their labels are defined.

Solvatochromic energy shifts are derived from paired acetone and gas-phase
predictions. does not train a direct wavelength-shift or intensity-shift
head. This removes the ambiguous and partially corrupted shift target from
the optimization objective.

## 3. Source data and exclusions

The source pool is the existing tmQMg*/tmQMg collection. Exact file and
directory identities are frozen in `manifests/source_data_manifest.json`.

Predeclared label handling:

- Nonfinite targets are masked only for the affected loss and metric.
- Wavelengths at or below zero are nonphysical and are converted to missing
  energy labels. The known negative acetone wavelength for YEJXOB is therefore
  excluded from that state-specific loss and metric.
- A regional band is present when tmQMg* reports its regional maximum under
  the `f_max >= 0.01` convention.
- Peak, CT, and NTO labels that are chemically undefined are missing, not zero.
- No direct `lambda_delta`, `f_delta`, or categorical shift flag is used for
  training, model selection, or a confirmatory claim.

No molecule may be removed because a model predicts it poorly.

## 4. Identity-aware split

The split seed is `20260724`. The input table is sorted by molecule identifier
before grouping and splitting.

Molecules are joined into the same identity group when any of the following
predeclared rules connects them:

1. Equal canonical isomeric SMILES and equal molecular charge.
2. Equal normalized raw SMILES and charge when RDKit cannot canonicalize the
   SMILES.
3. An exact geometry fingerprint or a verified near-duplicate geometry with
   matching elemental composition and a Kabsch RMSD no greater than 0.10
   angstrom, using the deterministic geometry procedure implemented in the
   tagged split code.

Connected components are used, so identity grouping is transitive. Missing
SMILES alone never joins unrelated molecules.

The grouped molecules are assigned by one deterministic 10-fold
`StratifiedGroupKFold` call with shuffling and the frozen seed. The
stratification label combines metal, charge, atom-count quintile, and the
gas/acetone visible-band availability pattern. A composite stratum containing
fewer than ten molecules cannot be represented across ten folds; all such
rare composite strata are therefore pooled under one deterministic label
before splitting. No molecule is discarded. Fold 0 is the locked test set,
fold 1 is validation, and folds 2-9 are training. The split is never rerolled.

Only predeclared structural checks may be run before training: complete and
disjoint membership, group disjointness, target sizes, and absence of
canonical/geometry identity overlap. The split IDs and hashes are frozen
before model training.

## 5. Model list

No architecture may be added to the confirmatory model ladder after this
freeze. The following previously motivated families are included:

1. XGBoost using the nine permitted ground-state descriptors.
2. Structure-only SchNet.
3. Structure-only PaiNN.
4. Graph-only baseline connectivity representation.
5. Graph-only u-NatQ.
6. Graph-only d-NatQ.
7. PaiNN fused with the baseline graph.
8. PaiNN fused with u-NatQ.
9. PaiNN fused with d-NatQ.
## 6. Fixed seeds and training budget

Five training seeds are deterministically derived from NumPy
`SeedSequence(20260724)`:

`2737188456`, `1409281222`, `2696188251`, `3663425383`, and `451851728`.

The common neural training budget is:

- AdamW optimizer, learning rate `2e-4`, weight decay `1e-5`.
- At most 300 epochs.
- Five-epoch linear learning-rate warmup.
- ReduceLROnPlateau factor `0.5`, patience 10 epochs.
- Early-stopping patience 40 epochs.
- Minimum validation improvement `1e-4 eV`.
- Gradient-norm clipping at `5.0`.
- Float32 training.
- Validation and early stopping use the joint molecule-level energy MAE across
  both solvents and all 30 states.
- SchNet, PaiNN, and NatQ variants use hidden size 192, five 3D interactions,
  three NatQ layers, conditioning size 256, dropout 0.1, cutoff 5.0 angstrom,
  and batch size 256.

The loss weights are frozen as:

`energy=1.0`, `oscillator=0.2`, `spectrum=0.2`, `band=0.2`,
`peak=0.2`, `CT=0.1`, `NTO=0.1`, and `state/peak consistency=0.05`.

Each component is averaged over its valid observations and solvents before
weighting. There is no direct shift loss and no aleatoric uncertainty loss.

XGBoost uses 300 estimators, maximum depth 6, learning rate 0.05, row
subsampling 0.8, column subsampling 0.8, and the same five fixed seeds. It is
trained separately for the two solvents. Training-only statistics are used
for preprocessing and any missing-target imputation.

No training extension or hyperparameter adjustment is allowed because a run
has not converged as hoped. Any protocol deviation is labeled exploratory.

## 7. Failed runs

A run is a technical failure only if one of these label-free conditions occurs:

- A nonfinite training loss or gradient is encountered.
- The selected validation checkpoint emits a nonfinite prediction.
- The selected validation checkpoint emits an excitation energy outside
  `(0, 15] eV`.
- The checkpoint cannot be read or does not match its frozen configuration.
- An infrastructure failure prevents creation of a scientific checkpoint.

High validation or test error is not a failed run. Test performance is never
used to exclude or replace a seed. Technical failures and logs are reported.
An infrastructure-interrupted run may resume or retry only with the identical
seed, configuration, data hashes, and software environment.

The ensemble is the arithmetic mean of all valid fixed-seed predictions. At
least four of the five seeds are required for a confirmatory ensemble claim.
No replacement seed is permitted.

## 8. Locked test policy

Training commands may read training and validation IDs only. They do not build
a test loader and do not write test metrics. The final evaluation command
requires an explicit unlock record and writes a permanent evaluation ledger.

The locked test set is evaluated once after:

- all fixed runs have terminated;
- run validity is decided using training and validation information only;
- all checkpoints, configurations, environments, and prediction rules are
  hashed;
- the ensemble member lists are frozen.

All preregistered models and valid seeds are evaluated together. No model,
seed, threshold, or aggregation rule changes after this evaluation can be
called confirmatory.

## 9. Endpoints and decision rules

### Primary endpoint

For each molecule, absolute energy errors are averaged over the 30 states and
both solvents. The primary endpoint is the mean of these molecule-level
errors for the five-seed ensemble. This gives each complex equal weight and
does not treat its 60 correlated predictions as 60 independent molecules.

### Confirmatory comparisons

The comparisons included in this package are:

1. PaiNN + d-NatQ fusion versus structure-only PaiNN.
2. PaiNN + d-NatQ fusion versus XGBoost.

For each pair, define the per-molecule difference as
`absolute-error(A) - absolute-error(B)`. Negative values favor A. Use 20,000
paired molecule-level bootstrap resamples with seed `20260724`. Report the
mean difference and percentile 95% confidence interval. A two-sided paired
bootstrap p-value retains the original three-test Holm correction. Exporting this
model subset does not change the reported adjusted p-values.

The statement "A outperforms B" is permitted only when:

1. A has at least `0.005 eV` lower primary MAE;
2. the paired 95% confidence interval excludes zero in A's favor; and
3. the Holm-adjusted p-value is below 0.05.

If the absolute MAE difference is below `0.005 eV`, the result is reported as
a practical tie even if the interval excludes zero. Other outcomes are
reported as inconclusive, not as evidence of equivalence.

### Secondary endpoints

Secondary results are descriptive and do not override the primary decision:

- Gas and acetone energy MAE, RMSE, and fractions within 0.05, 0.10, and
  0.20 eV.
- Per-state and per-molecule energy-error distributions.
- Oscillator-strength MAE in `log(1 + f)` and rank correlation.
- Normalized spectral similarity under the frozen broadening procedure.
- Band-existence AUROC, average precision, Brier score, expected calibration
  error, precision, recall, and F1 at probability 0.5.
- Conditional peak energy/intensity/width errors on truly defined bands and
  end-to-end gated coverage at probability 0.5.
- Visible CT macro-F1, balanced accuracy, and confusion matrix.
- NTO metal-fraction MAE.
- Statewise acetone-minus-gas energy-shift MAE derived from paired predictions.
- Deep-ensemble calibration and coverage after a single variance scaling
  parameter fitted on validation data.

NatQ representation ablations and SchNet/PaiNN differences are secondary.
They are reported with paired intervals but no unplanned superiority claims.

## 10. Reproducibility requirements

Every run records:

- Git commit and protocol tag.
- Dataset, split, configuration, cache, checkpoint, and software hashes.
- Exact seed and ensemble membership.
- Python and package versions, CUDA version, GPU type, SLURM job ID, hostname,
  start/end times, and command line.
- Counts of valid/masked targets and all run-integrity checks.

Graph-cache filenames and metadata include source, split, cutoff, NatQ schema,
and preprocessing hashes. Resume is refused when the current configuration or
input fingerprints differ from the checkpoint.

## 11. Reporting limitation

The new grouped test set comes from the same approximately 74,000-complex pool
that was explored during main. The numerical evaluation is protected from
test-result-driven choices, but the model families and scientific questions
were informed by main. The manuscript must state this distinction plainly.
