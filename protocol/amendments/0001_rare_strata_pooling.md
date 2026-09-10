# Protocol Amendment 0001: Rare Composite Strata

Date: 2026-07-24

Status when adopted: the train/validation/test split had not been generated,
no model had been trained, and the locked test set did not exist.

## Reason

The pre-split unit tests showed that a complete cross-product of metal, charge,
atom-count quintile, gas visible-band availability, and acetone visible-band
availability can contain categories with fewer molecules than the ten requested
folds. Such a category cannot be represented in every fold.

## Frozen Rule

The composite stratification label is constructed exactly as specified in the
main protocol. Composite labels represented by fewer than ten molecules are
pooled into one `pooled_rare_composite_strata` category. No molecule is removed,
no target value is changed, and no split is rerolled. This rule is applied
before the single seeded `StratifiedGroupKFold` call.

## Information Used

This amendment was prompted by a generic pre-split test warning. It was made
without training any model, without generating the split, and without
observing any test result.
