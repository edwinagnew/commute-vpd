# E307 (P2.66 run 4) Amendment 1: the step-300 replay check across cards is recorded, not a stop (2026-11-24)

**What was measured.** Pod 1 (RunPod L40S, $0.39) retrained arxiv s0, german s0 and spanish s0 to step 300 twice each.
On that card the two replays were bit-identical (max bf16 weight diff 0). The filed stage-0 control compared the
replay's read-row gain with the saved step-300 file, which was trained on a cluster L40:

| task s0 | replay gain | saved file gain | gap | filed floor (within one card) | result |
|---|---|---|---|---|---|
| arxiv | 0.51639 | 0.51393 | 0.00246 | 0.00077 | STOP |
| german | 0.69011 | 0.69433 | 0.00423 | 0.00143 | STOP |
| spanish | 0.95980 | 0.95967 | 0.00013 | 0.00219 | passed |

The replay's weights differ from the saved files by at most 1.1e-3. The floor measured only within-card variation, so
it could not cover a gap between an L40 and an L40S. Source: `experiments/convergence_panel/results/wide/run4/control_*_pod1.json`.

**Decision (whip-lead, 2026-11-24 ~20:0x, standing authority over small reruns).** The gap to the saved file is
recorded, not a stop. The stop is now: two replays on the pod's card that are not bit-identical. Both tolerances are
reported: the filed one (0.00077 / 0.00143) and the measured one across cards (0.0025 / 0.0042). Every arxiv and german
state used later comes from the new retrain, never from the old files. `predictions_run4.json` is unchanged. Code:
`run4.py` `control()` and `retrain()`'s step-300 check.

If arxiv or german never reach read-row gain 1 by step 1200, the recipe is not changed; the result goes to whip-lead,
who puts it to the user.
