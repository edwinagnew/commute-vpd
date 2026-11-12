# Plan: does reinforcement learning on Qwen stand apart from its shuffled-reward null at 2, 5 or 10 times the decomposition's error? (cell P3.4, run 2 of 3, E167)

## The question, in plain words
Run 1 found one recipe that grows the RL update while held-out accuracy and answer length hold: peak 3e-6 after 10
warm-up steps (2.65 times the masked error at step 150). The user chose it for run 2 (decisions.md, 2026-11-12, "do (a)").
Run 2 trains three arms of E69's GRPO loop on Qwen3-1.7B to step 400 on seeds 0 (draw seed 69) and 70: R (reward =
correct), N (shuffled rewards) and F (supervised training on W0's own sampled answers). It keeps each seed's states where
R's output change first reaches 2, 5 and 10 times the masked error (0.0932 nats per token), R, N and F at the same step, and
reads them all on one card type. Done when the cell's figure can say at which size, if any, R's held-out pass rate is
above N's by more than the seed spread.

## Design check (`/design`)
Fills the open cell P3.4 (owner qwen-rl-lead, run 2 of 3; brief `fleet/paper/briefs/2026-11-11_qwen-rl-readable.md`); the
recipe is the user's choice on card `reports/decisions/2026-11-12_decision_qwen-rl-run2-recipe.md`. No decomposition
reading beyond the step-0 tables the cell names; no reuse verdict.

## What runs
**Six H100 pods in parallel**, one arm and seed each (`R_s0 N_s0 F_s0 R_s70 N_s70 F_s70`), `experiments/qwen_rl_readable/run2.py`.
Every step is run 1's (`run.py` `train_step`, `read`, `drift_stop`, unchanged) through E69's `update`. At every 25th step
from 50 to 400 the pod writes a checkpoint and δW in fp32 (E69's layout, 5.64 GB) with its sha256 in its results JSON
(`saves`). R reads the output change every 5 steps and GSM8K every 50 (for the stop rule only); N and F read the output
change at their save steps. **F trains on fresh W0 rollouts at every step to 400**, drawn with R's problems and step
seeds, so F differs from R only in the reward (Fable's advice, 2026-11-12): E69's for seed 0 steps 1-100, sampled in the
loop otherwise. This costs ~$27 (6.8 pod-hours of sampling) more than reusing one draw per problem past step 114, and
makes F_s70 the longest pod.

**Matched states** (`match.py chosen`, filed): for each seed and size k, the first save step at or after R's first 5-step
read with multiple ≥ k; it is *confirmed* once a later GSM8K read of R held. R's log prints `CONFIRMED kx at step t`; the
runner then downloads step t from R, N and F of that seed (`run2.py --chosen` lists files and sha256). Nothing else is
downloaded. N and F may end once R has ended and they have saved R's last confirmed step.

**Stop rule** (filed, the brief's, run 1's `pick.drift` unchanged): at an R GSM8K read, pass below W0's 0.7817 or fits
below 0.9215 while the output change exceeds the previous GSM8K read's: R ends; its last state is the GSM8K read before.

**After the pods** (cluster L40, debug QoS, $0, `read2.py`): a controls job first (W0 and E69's saved R step 100), then
every confirmed state in 30-min jobs, one file each so jobs run side by side: GSM8K pass and fits (1319 problems, fp32),
output change and multiple, |δW| (`cluster/qwen_rl_run2_read.sbatch`); then the outside share against a Gaussian change of
the same norms and |δW| against the leftover norm 4.98 by E152's step-0 code (`cluster/qwen_rl_run2_share.sbatch`, CPU);
then `read2.py --stage verdict`, the filed "worked" rule.

## Predictions (`experiments/qwen_rl_readable/results/predictions_run2.json`)
- **Worked** (`match.worked`): at a size read on both seeds, R minus N on each seed above max(seed spread of either arm,
  2 paired se). Apart at any size: RL worked. Together at every read size: reported as the result. *Lean medium:* apart
  at 2x; 5x read on at most one seed; 10x not reached. At W0: together. E69's step-100 numbers as a stand-in: together.
- **Matched steps:** 2x near step 125 on seed 0; 5x around 335-410 or not by 400; 10x not reached.
- **Drift stop:** not expected (run 1's 3e-6 held to step 200).

## Cost and time (H100 $3.99/h; run 1's measured step times; 40 min set-up and 0.2 h download tail per pod)
| pod | per step (train + reads + saves) | hours | $ |
|---|---|---|---|
| R_s0, R_s70 | 65.9 s (53.3 step, GSM8K 388 s / 50, output change 5.5 s / 5, save+checkpoint ~100 s / 25) | 8.19 each | 32.7 each |
| N_s0, N_s70 | 59.3 s | 7.45 each | 29.7 each |
| F_s0 | 70.3 s (W0 sampling 42.5 s on steps 101-400) | 8.68 | 34.6 |
| F_s70 | 81.0 s (W0 sampling on every step) | 9.86 | 39.3 |
| **pods total** | wall ~9.9 h (F_s70) | 49.8 pod-h | **~$199** |
| downloads | 12 states x 5.64 GB / 10 MB/s = 6,765 s = 1.9 h (18 states: 2.8 h), overlapping the run | | in tails |
| cluster reads | 2 controls + ~12 states at ~15 min each on L40 (to measure), outside share ~1 h CPU | | $0 |

$199 is ~$29 over the ~$170 the user approved for run 2 (cell to ~$250 with run 1's $50.82, past the ~$220 approved),
because F's fresh W0 rollouts were not in the card's estimate. The lead asks the user before the pods start.

## Gates
Gated (>$60, >6 h). Pre-mortem, sourced settings, rehearsal and early checks: `reports/plans/2026-11-12_plan_qwen-rl-readable-run2-gates.md`.
Gate 5 (one fresh Opus review of plan and code) and gate 6 (the user's go-ahead with the rehearsal's numbers) are the lead's.

## Not in this run
No decomposition reading beyond the outside share and norms; no reuse verdict; no seed beyond 0 and 70; no rate other
than 3e-6w10; no longer window; no state other than the confirmed steps downloaded; no change to run 1's or E69's files.
