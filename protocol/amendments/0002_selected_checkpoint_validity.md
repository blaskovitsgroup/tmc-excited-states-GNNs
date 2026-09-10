# Protocol Amendment 0002: Selected-Checkpoint Validity

Date: 2026-07-25

Status when adopted: the locked test set had not been unlocked, loaded, or
evaluated. This correction was prompted exclusively by validation-only failure
records.

## Protocol Rule

Section 7 of the frozen protocol states that a seed is a technical failure when
the **selected validation checkpoint** emits a nonfinite excitation energy or
an energy outside `(0, 15]` eV.

## Implementation Mismatch

The initial trainer applied the `(0, 15]` range check after every intermediate
epoch. It therefore stopped eight seeds when a temporary intermediate
checkpoint crossed the range, even though those checkpoints had not been
selected as the best validation checkpoint. This was stricter than the frozen
rule and could prevent an otherwise valid run from converging.

## Correction and Retry Rule

Intermediate validation predictions must be finite, but a finite
out-of-range intermediate prediction does not terminate training. After
training, the trainer reloads the checkpoint selected by the frozen validation
MAE rule and applies the full finite and `(0, 15]` check to that checkpoint.

Only the eight seeds stopped by the original mismatch may resume. They retain
their original model, seed, data hashes, optimizer state, scheduler state,
hyperparameters, and maximum epoch budget. No replacement seed is introduced.
The original failure records and a retry ledger are retained.

## Information Used

The decision used only the failure messages and the already frozen protocol.
No locked-test ID, label, prediction, or metric was accessed.
