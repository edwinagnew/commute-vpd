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

## Correction (2026-10-14, after pod 3): the scope was the whole cause

The sections below this one record what was believed after pod 2 and are left as written. Pod 3 found that the
edit described under "Decision" had never reached the function body: the string replacement rewrote the module
docstring and left `faithfulness()` calling the authors' single-batch `pgd_recon_loss`, one adversary per 8-row
batch. Pod 2's "multi-batch" calibration therefore re-measured the same per-8-row adversary (8.39 against 8.11). With
the multi-batch call actually in place (commit `ea4df42`, one adversary shared across 128 rows, gradients
accumulated over 16 sub-batches), the released decomposition on W0 gives 0.68 and 0.72 on two 128-row blocks against
the paper's 0.927 and the 0.59 to 2.24 that the paper's run logged over 100 random batches. The authors' function
and an independent re-implementation agree bit for bit. The code-identity check and the W&B history below stand;
the "second calibration" paragraph's conclusion that scope was not the cause does not. The paper's adversarial
number is reproducible.

## Erratum (2026-10-14, code audit after round 1)

An audit of the commits and pod logs (`results/large/logs/`) corrects four statements in this report and one made in
chat:

- **Only the scope of the adversary moved the number.** In chat, the session's lead agent said "a grad-disabled
  decorator corrupted the PGD attack". That was wrong. `faithfulness()` carried `@torch.no_grad()`, but the authors'
  single-batch PGD loop enables gradients internally, so the per-8-row runs (8.11, 8.39) were correct per-8-row
  numbers. The multi-batch helper has no such guard: under the decorator it raises an error
  (`faith_gradmode.log`), it does not return a wrong value. Commit `ea4df42` made grad mode explicit so that the new
  call could run at all; that change did not lower the number.
- **"Second calibration as read at the time", below.** These statements are false as written:
  - "With one adversary shared across 128 rows the metric came out at 8.39". That run used one adversary per 8 rows.
  - The step table (2 / 5 / 10 / 20 → 0.33 / 0.84 / 2.86 / 8.39). It compares per-8-row numbers with the paper's
    128-row table.
  - "What is left is the rows themselves … or the environment". Neither was the cause.
- **Commit `6773167`'s message** says the code switched to the multi-batch PGD. Only the module docstring changed.
- **"Agree bit for bit" above is overstated.**
  - They agree bit for bit on block 0 (0.68064266). On block 1 they differ in the 7th digit.
  - The "independent" loop (`check_pgd.py`) reuses the authors' mask-construction and initialisation helpers. Only
    the loop and the gradient accumulation are independent.

None of these change a scored result. The current path is covered by `tests/test_faithfulness_invariants.py`, which
fails on the code at `d0ed6c5`.

## Second calibration as read at the time (superseded by the correction above)

With one adversary shared across 128 rows the metric came out at 8.39 (stochastic 0.245, CI-masked 0.344, unmasked
0.012 unchanged). Against the number of PGD steps on the same rows: 2 → 0.33, 5 → 0.84, 10 → 2.86, 20 → 8.39; the
paper's table gives 20 → 0.83, 40 → 1.35, 80 → 3.84, 160 → 25.3, 320 → 40.2, and zero-ablating everything gives
67.4 in both. Three further checks:

- The paper's run (`s-55ea3f9b`, created 2026-02-25) was trained at commit `85c6b70`. Its PGD code path
  (`spd/metrics/pgd_utils.py`, the mask interpolation, the loss normalisation, the eval wiring) is functionally
  identical to the pinned commit `74146b5`: the diff is renames and the loss-function plumbing.
- W&B logged `eval/loss/PGDReconLoss` 401 times during that run (one random 128-row val batch each, 20 steps of
  0.1, shared across the batch): 0.652 at the final step, mean 0.82, min 0.59, max 2.24 over the last 100k steps.
  The paper's 0.927 is in that range. So the paper's number is real and batch-noisy by a factor of 2 to 4.
- Our 8.4 on val rows 0..127 is far outside that range. What is left is the rows themselves (a heavy-tailed
  per-position KL could make one block atypical) or the environment (torch 2.8 kernels, one GPU rather than their
  DDP). `check_pgd.py` measures blocks 1..3 and the per-position distribution on block 0; its result goes in the
  analysis report.

Decision, second part: the pipeline is not held for this. Prediction 3 is a paired within-harness ratio on the same
rows and seed; the absolute level of the adversarial metric does not enter it. Faithfulness is averaged over two
128-row blocks (val rows 0..255), one shared adversary per block, per-block values reported, so that a single
atypical block cannot carry the number. Prediction 4's gate is read on the stochastic KL (which matches), and the
adversarial level is reported with this diagnosis beside it.

## Reading rule for the adversarial half of prediction 3 (fixed before the diagnostic reports)

Written 2026-10-14 after pod 2 and before pod 3's `check_pgd.py` numbers were seen. Prediction 3 has two ratios,
stochastic and adversarial. The adversarial ratio is scored as filed only if, on W0, (a) the two 128-row blocks'
20-step adversarial KL agree within a factor of 2 and (b) the top 1% of positions carry less than half of the mean
in each block. If either fails, the adversarial ratio is reported but not scored and prediction 3 rests on the
stochastic ratio alone. Predictions 1a, 1b and 2 do not use the adversarial metric. (Written when the gap
was thought open; see the correction above: the gap closed once the scope change actually reached the code.)

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
