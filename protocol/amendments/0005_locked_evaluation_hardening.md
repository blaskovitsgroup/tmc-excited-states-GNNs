# Amendment 0005: Locked-Evaluation Implementation Hardening

Date: 2026-07-26

## Status at the time of this record

The locked test set had not been evaluated or loaded. Review of the final
evaluation implementation identified missing output checks and a missing
implementation of one already frozen secondary endpoint.

## Corrections

The locked-prediction and summary stages now:

- verify the exact shape and finiteness of every prediction array;
- verify probability-valued outputs remain in `[0, 1]`;
- isolate each parallel model evaluation in a model-specific graph-cache file
  to prevent concurrent cache creation or overwrite;
- refuse a one-time submission when locked predictions or results already
  exist, and persist the submitted job IDs with the frozen input hashes;
- distinguish a fully resolved model with technical seed failures from an
  ensemble that is still preliminary because runs are missing or incomplete;
- cross-check model identity, fixed-seed membership, eligibility, ensemble
  manifest hash, unlock-record hash, prediction hash, and metadata hash;
- refuse to let a nonfinite prediction be silently omitted by a mask-aware
  metric;
- serialize unavailable descriptive metrics as JSON `null` rather than the
  nonstandard tokens `NaN` or `Infinity`; and
- report the already frozen deep-ensemble uncertainty endpoint using raw
  member variance and the single scalar variance multiplier fitted on
  validation data before test unlock. Reported quantities are state-energy
  Gaussian NLL and empirical coverage at one and 1.96 standard deviations.

In addition, every valid neural checkpoint is replayed once on the validation
partition before ensemble freezing. This audit checks every output head for
finiteness and its architectural constraints, then confirms that the replayed
state energies reproduce the immutable validation prediction artifact for the
same checkpoint and seed.

These corrections add no model, seed, endpoint, threshold, calibration fit, or
comparison. They make the final implementation enforce and report rules that
were already specified in the frozen protocol.

## Test-set statement

The corrections were designed and tested with synthetic arrays and
training/validation artifacts only. The locked test set remained unopened and
contributed no information.
