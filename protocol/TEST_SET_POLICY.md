# Locked Test-Set Policy

## Why the test set is locked

The training set teaches model parameters. The validation set is used for
early stopping and other decisions that were frozen in advance. The test set
answers the final question: how well does the already-finished procedure work
on molecules it did not use for learning or decision-making?

Looking repeatedly at test error turns the test set into another validation
set. This can happen even without feeding test labels into gradient descent.
For example, replacing a seed because its test error is high is test leakage.

## rule

training code must not instantiate a test data loader or calculate a test
metric. The test IDs are stored separately from the training/validation split.
Final evaluation requires a deliberate unlock file created only after the run
manifest and ensemble membership are frozen.

The evaluation ledger records when the test set was unlocked, the commit,
input hashes, checkpoints, command, and output hashes. It is append-only. A
second evaluation may be performed for debugging or exploratory work, but it
must be labeled non-confirmatory.

## What may be checked before unlocking

The following checks do not use model performance and are allowed:

- Every molecule appears in exactly one partition.
- Identity groups do not cross partitions.
- Gas and acetone records remain paired.
- Required files exist and hashes match.
- Test targets satisfy the same general physical-integrity rules defined
  before splitting.

No test prediction, test loss, test MAE, or test-based threshold selection is
allowed before the final evaluation.
