# Plan: does reinforcement learning on Qwen stand apart from a working shuffled-reward null of the same output change? (cell P3.4, run 3 of 3, E178)

## The question, in plain words
Run 2 (E167) read reinforcement learning (RL) at peak 3e-6 against its shuffled-reward null at the same step, and the
null collapsed: its held-out pass rate fell to 0.04 and 0.00, so "apart" said nothing. The user left the last run to the
lead, who chose option (a) of `reports/decisions/2026-11-12_decision_qwen-rl-last-run.md` (decisions.md, 2026-11-13,
"P3.4 last run"). Run 3 trains only the null, at peak 1e-6, the rate where E69's and run 1's nulls held for 100 steps,
on seeds 0 and 70, and stops each where its output change reaches that of run 2's RL state at step 75 on the same seed.
That RL state is already saved and read. Done when the cell can say whether RL's held-out pass rate at about two to
three times the decomposition's error is above a working null that changes the output as much, by more than the seed
spread, or that no working null was found.

## Design check (`/design`)
Fills cell P3.4 (owner qwen-rl-lead, run 3 of 3), option (a) as recorded in decisions.md. Two departures from the cell
as written, both on the card the user delegated: the null trains at a third of RL's rate, and it is matched by output
change, not by step. No decomposition reading, no reuse verdict. Card `experiments/qwen_rl_readable/card_run3.txt`
(passes: the smallest gap that matters, 0.03, is 3.05 paired standard errors at n = 1138).

## What runs
**Two H100 pods in parallel**, `N_s0` and `N_s70`, `experiments/qwen_rl_readable/run3.py --pod <pod>`. Each step is run
1's `train_step` through E69's `update` with the shuffled reward, E69's recipe `1e-6w10` unchanged (64 problems x 8
samples, window 512, AdamW, fp32). The output change is read every 5 steps on the pod (run 1's `read`), a checkpoint is
written every 25. The pod ends at the first of:
- **collapsed** (filed): 5 steps in a row each with under half its training answers right or mean answer length under 100
  tokens. No state is saved. The three nulls at 1e-6 never went below 0.53 and 125 tokens (verified, steps 1-100); the
  rule fires at step 31 and 26 on run 2's collapsed nulls.
- **matched**: the first read whose multiple of the masked error (0.0932 nats per token) is at least the target of its
  seed: **2.0882 on seed 0, 3.0570 on seed 70**. δW is saved there and downloaded.
- **cap**: step 500 or 8.3 h of training and reads. δW is saved for a descriptive read only.

**The yardstick.** The targets are run 2's RL pods' own 5-step reads at step 75 (`results/run2/R_s*.json`
reads["75"].multiple), the same function and rows the null's pod reads with. The verdict compares the step-75 state, so
the null is matched to that state's size. For seed 70 that is 3.06, not the 2.42 of its crossing read at step 55: that
was a different state. The pod and the L40 agree on these states to 1e-7 (L40 2.0881820 and 3.0569943), so 3.06 is
no card effect.

**After the pods** (cluster L40s, debug QoS, $0, read2.py's code through `read3.py`): the controls first and alone (W0 and
E69's R at step 100, run 2's bands), then each matched state (output change, greedy GSM8K pass and fits on 1319 problems,
|δW|), then `read3.py --stage verdict`.

## Expected values
| | W0 (do-nothing) | control: E69 step 100 as stand-in | rival A (reward matters) | rival B (own answers do it) |
|---|---|---|---|---|
| null's multiple at its end | 0, never matched: "cap" | 0.78 at step 100 (run 1's null) | matched, about target + 0.02-0.05 | same |
| null's pass (L40) | 0.782 | 0.870 / 0.794 | under RL by more than the margin on both seeds, e.g. 0.75 | within the margin on a seed |
| verdict | "no call" (no matched state) | "together", margin 0.076 | "apart" | "together" |

R reads 0.8816 (seed 0) and 0.8844 (seed 70) on the L40, already.

## Predictions (`experiments/qwen_rl_readable/results/predictions_run3.json`, filed 2026-11-13)
- **Verdict** (`match3.verdict`, run 2's `match.size_call` at one size): gap = R − N per seed; margin = max(seed spread of
  either arm, 2 x the larger paired standard error); "apart" if gap > margin on both seeds, else "together"; "no call"
  if a seed's null collapsed, was capped, or a number is missing. *Lean: "together", low confidence*: E69's working nulls
  alone differed by 0.076 between seeds, and run 1's null read 0.778 against E69's 0.870 on the same seed and recipe.
- **Collapse**: not fired on either seed, medium confidence. The risk is real: run 2's nulls collapsed near |δW| 0.5,
  which this null may reach near step 190.
- **Match steps**: seed 0 between steps 150 and 400 (central ~250); seed 70 between 200 and 500 or not by the cap
  (central ~330). Low confidence: the null has never run past step 100 at 1e-6.
- **Early checks** (stops): step-1 reward in [0.6, 0.88]; the first step's |δW| ratio and KL on H in E69's bands;
  reward change by step 20 above −0.05; |δW| at step 20 in [0.5x, 2x] of E69's null of the same seed; seed 0's multiple
  at step 100 in [0.5825, 1.0489]. Logged: step-1 rollouts equal E69's, steps 1-5 reward, length and |δW| as E69's null.
- **Reported beside the verdict**: whether the null's pass stays at or above 0.7597 (W0's band) on each seed.

## Cost and time (H100 $3.99/h; 40 min set-up; ~62 s a step)
Step time: run 1's null at 1e-6 averaged 55.3 s on an H200 (`results/N.json` steps); run 2's nulls 53.9 and 60.7 s over
steps 1-25 on H100 before their collapse (their whole-run mean, 12 s, is the collapsed null's short answers); plus the
output change (5.5 s every 5 steps) and the checkpoint (15 s every 25). E69's null's answers grew from 188 to 219 tokens
over steps 1-100, so later steps may run slower (E69's mean 77 s on its card); the hours cap bounds the spend whatever
the step time.

| stage, per pod | expected seed 0 | expected seed 70 | cap |
|---|---|---|---|
| set-up (venv, model, W&B fetch of H) | 0.67 h, $2.7 | 0.67 h, $2.7 | 0.67 h, $2.7 |
| steps 1-20, the early checks (counted in the next row) | 0.34 h, $1.4 | 0.34 h, $1.4 | 0.34 h, $1.4 |
| training to the end | ~250 steps, 4.3 h in all, $17.2 | ~330 steps, 5.7 h in all, $22.7 | 8.3 h, $33.1 |
| download one δW (5.64 GB / 10 MB/s = 9.4 min), JSON, terminate | 0.26 h, $1.0 | 0.26 h, $1.0 | 0.26 h, $1.0 |
| **pod** | **5.2 h, $21** | **6.6 h, $26** | **9.2 h, $36.8** |

Two pods: **~$47 expected, $73.7 at most** (both capped), under the $75 approved. A repeat collapse ends a pod within a
few steps of its start: ~$5-15 a pod. Cluster: 2 control reads and 2 state reads of ~16 min (run 2's 918-960 s) on L40s,
$0. The cell comes to ~$300 at most with run 2's ~$226.

## Gates
Gated (>6 h a pod, >$60 at most). Gates 1-4 in `reports/plans/2026-11-13_plan_qwen-rl-readable-run3-gates.md`: pre-mortem,
sourced settings, the $0 rehearsal (`cluster/qwen_rl_run3_rehearsal.sbatch`, numbers measured 2026-11-13, in the appendix) and the early
checks. **Gate 5, review: done 2026-11-13** (fresh Opus, plan and code together): no blockers. Taken up: the worst case is ~$77, not $73.7 (the hours cap counts sampling, training and reads, not ~0.5 h of checkpoints and saves, plus billing to terminate); a restart after a lost pod, a band stop, or one seed collapsing goes to the lead before more spend (pod-runner brief); N over R multiple is computed in the analysis. Not changed: no early end for a seed on course to miss (the design is fixed). **Gate 6, go-ahead: given 2026-11-13** (user, dialog: "Run it"; ~$47 expected, ~$77 worst, ~6 h).

## Not in this run
No RL or fine-tune arm; no rate other than 1e-6w10; no third seed; no GSM8K read on the pod; no decomposition reading
(outside share, norms); no state other than the matched or capped one downloaded; no change to run 1's or run 2's files.
