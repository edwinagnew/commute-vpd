# Decision: the adversarial faithfulness metric uses one adversary across all evaluation rows

*2026-10-14, coding agent. A measurement-apparatus change made after `predictions.json` was filed (12:49) and
before any scored measurement was taken. The filed file is not edited; this report records the deviation.*

## What happened

The calibration stage (prediction 4, not scored) reproduced the released decomposition's numbers on the unmodified
model W0 for everything but the adversarial KL:

| measure | ours on W0 | reference | ratio |
|---|---|---|---|
| alive subcomponents per layer | 3625 / 844 / 1940 / 3472 (9881) | paper 3709 / 848 / 1943 / 3472 (9972); vpd-audit 9959 | agrees |
| stochastic reconstruction KL | 0.245 | paper 0.238 | 1.03 |
| CI-masked KL | 0.344 | vpd-audit 0.342 | 1.01 |
| unmasked KL | 0.0110 | vpd-audit 0.0115 | 0.96 |
| adversarial (PGD) KL | 8.11 | paper 0.927 | 8.75 |

A read-only diagnostic on W0 (`results/large/logs/pgd_diagnostic.json`) showed the PGD number depends on how many
rows one adversary has to cover: 7.7 at 8 rows per adversary, 4.0 at 16, with Δ inclusion and precision irrelevant
and 0 steps giving 0.26 (the stochastic value). The filed set-up ("batches of 8 ... one adversary per batch") gives
each adversary 8 rows; the paper's evaluation (`PGDReconLoss`, `mask_scope: shared_across_batch`, eval batch 128)
gives it 128. A 24 GB card cannot hold 128 rows of causal importances at once, which is why the filed set-up chose 8.

## Decision

Use the authors' multi-batch PGD (`calc_multibatch_pgd_masked_recon_loss`: one adversary shared across every row,
gradients accumulated over sub-batches of 8 before each of the 20 steps of 0.1, random init, Δ included), on Pile
val rows 0..127, so that the adversary's scope is the paper's 128 rows. The stochastic, CI-masked and unmasked
metrics are computed on the same 128 rows (previously 64). Nothing else changes.

## Why this is a repair and not a post-hoc choice

- Prediction 4 exists exactly to catch this: "if not [within 1.5×], the harness is wrong and nothing else is
  scored until it is fixed". No scored measurement had been taken.
- The change makes the metric match the paper's definition; it does not touch any threshold or rule in
  `predictions.json`. Prediction 3 is a within-harness ratio and would have been scorable either way, but with a
  per-8-row adversary its absolute level would not have been comparable to the paper's.
- Prediction 4 is re-evaluated under the repaired metric; the result is recorded in `results/calibration.json`.

## Trace

The failed-gate file was moved to `results/large/logs/calibration_per8row_adversary.json` (gitignored; its numbers are
in the table above); the rerun writes a fresh `results/calibration.json`. The diagnostic is `results/large/logs/pgd_diagnostic.json`.
