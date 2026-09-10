# Amendment 0006: Post-Unlock CT Index Type Correction

Date: 2026-07-26

## Status at the time of this record

The one-time locked prediction jobs had completed, but the dependent summary
job failed before writing any result file or metric. The traceback showed that
masked charge-transfer (CT) class targets were stored as floating-point values
and passed directly to a NumPy class-name lookup, which requires integer
indices.

No locked-test metric, model comparison, figure, or table had been produced or
observed when this correction was made. All 11 immutable prediction archives
and their metadata remained intact.

## Correction

For masked CT targets only, the summary code now:

- verifies that every class index is finite;
- verifies that every class index is integer-valued;
- converts the verified values to integer indices; and
- verifies that each index is within the frozen four-class CT vocabulary.

A synthetic regression test now represents CT targets with the same
floating-point storage type and confirms successful end-to-end summary
generation. The complete local test suite passes after the correction.

Only the failed summary job may be rerun. No training, checkpoint selection,
ensemble membership, prediction, calibration, endpoint, threshold, comparison,
or statistical rule is changed.

## Test-set statement

The correction was determined solely from the exception type and traceback.
It is a data-type compatibility repair required to execute the already frozen
analysis. It did not use any test target value, prediction error, aggregate
metric, or comparison outcome.
