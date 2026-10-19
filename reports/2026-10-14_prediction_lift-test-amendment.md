# Lift test, amendment 1: predictions 2′ and 3′

*2026-10-14, coding agent, at the user's request ("file and run"). Filed in
`experiments/lift_test/results/predictions_amend1.json` before either run. `results/predictions.json` is unchanged, and
its predictions stay scored as filed: P1 reversed, P2's L_F term unscorable, P3 failed. Results so far:
`reports/2026-10-14_analysis_lift-test.md`.*

## Why amend

- **P2.** Its weight term L_F is blind. ‖Δ₀‖² = 67.9, so even the do-nothing map scores 1.004. Its masked KLs do not
  separate maps either, because do-nothing already passes them (stochastic 0.99, PGD 1.04): Δ absorbs δW. The term
  that does separate maps is the unmasked KL with Δ excluded. That is the components alone, run unmasked, which shows
  whether they carry the fine-tuned model. Do-nothing scores 1.85 on it at step 1.
- **P3.** The filed learning-rate grid moved the weights 3–100× more than the W fine-tune does, which breaks the
  small-update assumption A2. The selection rule also ignored off-target damage.

## P2′: the Euclidean lift is faithful and carries the update

- **Updates.** The arm A rerun at checkpoints 1, 2, 3, 5 and 10.
- **Map.**
  - The minimum-norm lift of δW onto the alive U, V (tolerance 1e-6).
  - Target weights W₀ + δW, the real fine-tuned model, so Δ′ = Δ₀ + O(δW) − dU dVᵀ.
  - Compared against do-nothing at the same checkpoints and against W₀, all in one job and one harness.
- **Gap recovered.** (r_nothing − r_lift) / (r_nothing − 1) on the unmasked KL with Δ excluded, where r is the ratio
  to W₀.
- **Pass.** At every scorable checkpoint, the lift's stochastic and PGD ratios are ≤ 1.5 and the gap recovered is
  ≥ 0.5.
- **Scorable.** A checkpoint counts only if r_nothing − 1 ≥ 0.2 on that KL.
- **Reported.** CI-masked ratios, and ‖Δ′ − Δ₀‖² / ‖δW‖².
- **Prediction.** Pass.

## P3′: pull-back at a matched step size

- **Training.** Adam on the alive (U, V) offsets, with arm A's loss, windows, batch, clip and seed. Learning rates
  1e-5, 2e-5 and 3e-5, 30 steps each.
- **Selection.** Among learning rates with held-out P(o) ≥ 0.9 at step 30, take the one with the lowest held-out KL
  elsewhere in the window at step 30. If none qualifies, take the one with the highest P(o).
- **Pass.** All three of:
  - held-out P(o) ≥ 0.9 by step 10;
  - held-out KL elsewhere at step 10 ≤ 0.110, twice the W fine-tune rerun's 0.0550;
  - stochastic and PGD ratios ≤ 1.5 at step 10.
- **Reported.** Faithfulness at step 30, and the pull-back's ‖δW‖ / ‖W₀‖ beside the W fine-tune's.
- **Prediction.** Pass.
