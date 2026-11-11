# Plan: which way of growing the RL update reaches 10 times the masked error while the pass rate holds? (E163, cell P3.4 run 1)

## The question, in plain words
The saved reinforcement-learning (RL) updates on Qwen3-1.7B change the model's output by only 0.7 to 1.05 times the
decomposition's own masked error (0.0932 nats per token, E152 step 0), too small to read. Run 1 first reproduces E69's
GRPO loop at its chosen recipe (R, the shuffled-reward null N and the own-samples fine-tune F, seed 0, peak 1e-6 after 10
warm-up steps, 100 steps), then grows R's update three ways (more steps at 1e-6, peak 3e-6, peak 1e-5) and picks the
recipe that reaches the largest multiple of the masked error (10, 5 or 2 times) while its held-out GSM8K pass rate stays
within 0.03 of R at step 100 and above the unchanged model's, with as many answers fitting the window as the unchanged
model's less 0.03. Run 2 then trains R, N and F on two seeds with that recipe.

## Design check (`/design`)
Fills the open cell P3.4 (owner qwen-rl-lead, run 1 of 3, the cell brief `fleet/paper/briefs/2026-11-11_qwen-rl-readable.md`):
part 3 needs an RL update above the decomposition's noise floor before any reading. Nearest TRIED row E69 (the loop, sizes
0.27-0.37); this adds the recipe that makes the update large enough, and no reading of the decomposition (not in the cell).

## States
Every arm starts from W0 (Qwen3-1.7B, fp32, TF32 off) through E69's code, unchanged (`experiments/qwen_rl/run.py`
`update`, `start_arm`; `job.py` `Job`), loaded by `experiments/qwen_rl_readable/run.py`. Differences from E69's end state:
none for the reproduction to step 100 (same seed 69, prompt order, H, sampler, Adam, rate schedule); past step 114 the
7345 training problems are used again in E69's order with new draws (step seed counts on); the 3e-6w10 and 1e-5w10 tags
are new (`src/commute/qwen_rl.py` RATES, two keys added; E69's run.py, job.py, fetch.py and tests still pass, 72/72).
No uploads, no δW saved; one checkpoint (~17 GB, pod disk only) at each checkpoint step, written right after that step's
update and before its read, so a crash in a read replays only the read.

## Pods (RunPod H100 80GB HBM3, in parallel; set-up and launch commands in the appendix below)
| pod | arm, rate | steps | GSM8K reads (and checkpoints) | early stops (exit 2), from `results/predictions.json` |
|---|---|---|---|---|
| 1 control (first, alone) | R, 1e-6w10 | 250 | 25 50 100 150 200 250 | E69's step-1 and first-live checks; reward change by 20 > -0.05; step 20 \|δW\| in [0.1296, 0.1433]; step 100 \|δW\| in [0.333, 0.368], pass in [0.8587, 0.9027] |
| 2 N | N, 1e-6w10 | 100 | 100 (checkpoints 20 50 100) | E69's checks; step 100 \|δW\| in [0.3144, 0.4253] (pass logged against 0.870) |
| 3 F | F, 1e-6w10 | 100 | 100 (checkpoints 20 50 100) | E69's checks (no reward change); step 100 \|δW\| in [0.260, 0.288], pass in [0.7832, 0.8272] |
| 4 r3e-6 | R, 3e-6w10 | 200 | 10 25 50 100 150 200 | step-1 and first-live checks (KL band scaled to the rate), finite loss; step 20 \|δW\| in [0.2047, 0.8187]; drift stop |
| 5 r1e-5 | R, 1e-5w10 | 100 | 10 25 50 100 | as pod 4; step 20 \|δW\| in [0.6822, 2.729] |

The R pods also read the output change alone (KL and \|δW\|, no GSM8K) every 5 steps, written to the JSON and the log as
each finishes; the selection uses only steps with a GSM8K read. N's \|δW\| band is ±15% of 0.370 because N's own seed
spread is wide (E69 seed 0 vs 70 at step 100: \|δW\| 0.3698 vs 0.3354, pass 0.870 vs 0.794). The step-20 bands on pods 4
and 5 are [0.5, 2] times E69's R step-20 \|δW\| (0.1364) scaled by peak / 1e-6: they catch a rate not applied, not a
slightly different trajectory.

**Stage gate.** Pods 2-5 start only after pod 1 logs this exact line, printed once its step-25 read is logged with a finite
output change and pass rate and its step-25 checkpoint is on disk (about 1.3 h after its start, set-up included):
```
GATE OPEN: control step 25 read logged with finite kl and pass, checkpoint written; start pods 2-5
```
Any stop in pod 1 before that: nothing else starts; tell the lead. If the control misses a band at step 100 after pods 2-5
have started, they run on and their reads stand as recipe evidence; the miss goes to the lead before anything is read.

At each read (written to `results/<pod>.json` and logged as it lands, so the runner downloads as it goes): \|δW\|; the output
change, KL(W_t ‖ W0) per token over every position of the 128 maths rows at length 256, the quantity of E93's
`clean.W0.K` (`qwen_d3.maths_rows`, `vpd.kl_per_position(updated, W0)`, batched since the rows have no padding), and its
multiple of 0.0932; at a GSM8K read also the greedy GSM8K pass rate (`qwen_eval.score`, strict_dropped, 1319 problems), the
share of answers that fit the window (`fits_share`) and the paired gain over W0. Expected at W0: 0, 0, 0.7817, 0.9515.
Positive control for the output change: the control at step 100 should read about 0.0643 (E93 on E69's saved step-100
update; band 0.058-0.071, logged, not a stop: a miss goes to the lead before the sweep is read). Blind to: the output
change pools every maths position, so it cannot say where the change is; the pass rate is greedy only, and counts only
answers that fit the window, which is why the fits share is a bar beside it.

## Predictions (filed in `experiments/qwen_rl_readable/results/predictions.json`)
- **Reproduction.** The bands above (control and F: E69 seed 0 ±5% for \|δW\|, measured spread L40 vs H100 0.013 and seed 0
  vs 70 0.004; pass ±0.022, 2 binomial se; N: \|δW\| ±15%, pass logged). At W0 every stopping band misses; at E69's own values
  every band holds. *Lean: high*, same code, seed and card.
- **Selection** (`pick.py select`): bar = max(control's pass at 100 − 0.03, W0's 0.7817); a GSM8K read is eligible at pass ≥
  bar and fits ≥ W0's 0.9515 − 0.03 = 0.9215; its level is the largest of 10, 5, 2 its multiple reaches. A recipe stopped on
  a check has no eligible read; the choice is made among the rest. Chosen: largest level, then fewer steps, higher pass,
  lower rate. Level ≥ 5: chosen; else "not reached" (RL could not get there at these recipes; run 2's design goes back to
  the lead); a recipe neither finished nor stopped, or no control read at 100: "incomplete". Edge cases (results below)
  each gave the intended outcome. Expected: the control alone about 0.7x at 100 and 1.5-2.5x at 250 (level 0 or 2).
  *Lean: medium* that 3e-6 reaches 5x or 10x above the bar by step 200; the card's two rivals are the readings.
- **Drift stop** (`pick.py drift`, R pods, GSM8K reads): pass < 0.7817 or fits < 0.9215, while the output change is above
  the previous GSM8K read's (0 before the first), ends the arm there; the GSM8K read before is its last good one. Not
  expected on the control (pass ~0.88, fits ~0.95 in E69); possible on 1e-5.

Edge cases (`~/commute-data/scratch/qwen-rl-lead/e163-fixer/edge.py`, 2026-11-11): expected shape → chosen r3e-6 10x at 200;
no reads → incomplete; missing pass → not reached; all under the bar → not reached; tie level and step → higher pass;
tie level, step and pass → lower rate; tie level → fewer steps; exactly 10x exactly at the bar → chosen; W0 above the
control's bar → bar is W0's; only 2x → not reached; one recipe unfinished → incomplete; no control read at 100 →
incomplete; W0 everywhere → not reached; r1e-5 stopped → chosen among the rest (r3e-6 5x); both sweep recipes stopped →
not reached; control stopped at its step-100 band → its read still sets the bar; fits just under its bar → not eligible;
fits exactly at its bar → eligible; fits missing → not eligible; 10x on an output-change-only read → not eligible; best
read at GSM8K step 10 with output-change reads around it → chosen at 10. Drift: fires below W0 growing, on the first
read below W0, and on fits under its bar growing; not when shrinking, at W0 exactly, fits exactly at its bar, any number
missing, or on an output-change-only read.

## Cost (H100 $3.99/h; E69's measured H100 times: R step 56.4 s, N 76.9 s, F train 34.4 s from `lr1e-6w10/{R,N,F}.json`
steps; GSM8K read ~420 s from `lr1e-6w10/run.log` step 100 read to `[accuracy R gsm8k]`; set-up 40 min per fresh Qwen pod;
output change ~30 s and checkpoint ~60 s per read are not measured)
| pod | GSM8K reads, output-change-only reads, checkpoints | hours | $ |
|---|---|---|---|
| 1 control | 6, 44, 6 | 5.1 + 0.7 set-up = 5.8 | 23.3 |
| 2 N | 1, 0, 3 | 2.3 + 0.7 = 3.0 | 12.0 |
| 3 F (E69's F rollouts fetched; sampled instead: +1.3 h, +$5.2) | 1, 0, 3 | 1.1 + 0.7 = 1.8 | 7.3 |
| 4 r3e-6 (steps +20% for longer answers) | 6, 34, 6 | 4.9 + 0.7 = 5.6 | 22.3 |
| 5 r1e-5 (steps +20%) | 4, 16, 4 | 2.6 + 0.7 = 3.3 | 13.1 |
| total | wall clock ~7 h (gate at ~1.3 h, then r3e-6's 5.6 h) | 19.5 | ~78 |

## Gates
Gated by the $60 line (~$73+). Go-ahead: the user's task file 2026-11-11 allows up to $200 on RunPod for this cell.
Rehearsal: on the paid card, as the control's first stage, stopping before pods 2-5 start (gate 3's paid-card clause).
Review: fresh Opus, 2026-11-11; findings applied (stage gate at the step-25 read and checkpoint; checkpoint before the
read; N's \|δW\|-only band; the fits bar; output-change reads every 5 steps and GSM8K reads at step 10; checkpoint steps
apart from reads; step-20 \|δW\| bands on pods 4-5; a stopped recipe has no eligible read; this section) or noted. Session
budget $300; the run is ~26% of it.

Pre-mortem (gate 1), with the guard for each:
- **Recipe.** E69's loop and recipe unchanged; 3e-6 and 1e-5 are the swept variable, not sourced settings. Guard: E69's
  step-1, first-live and step-20 checks; the step-20 \|δW\| bands catch a rate not applied.
- **Data.** E69's split, prompt order, chat template and window; past step 114 problems repeat with new draws (stated).
  Guard: step-1 reward band; rollout sha logged against E69's.
- **Precision.** fp32, TF32 off, as E69. Guard: finite loss and gradient each step; finite kl and pass at the gate.
- **Memory and host RAM per stage.** E69 peaks (`lr1e-6w10/*.json` `peaks`): R and N 69.4 GB on the card, F 52.5 GB; host
  10.8-11.3 GB, checkpoint writes included. The output-change read adds two 0.6 GB logit copies plus the KL's
  intermediates (~3 GB) beside Adam, never run on a GPU: it runs first at step 5 on every R pod (6 min in). GSM8K reads use
  E69's `score` at batch 128, which E69 ran with the trainable copy and Adam held.
- **Time and cost per stage.** Table above; the KL read and checkpoint times are not measured: the control's step-5 and
  step-25 log lines measure them before pods 2-5 start.
- **What is saved, where, size.** `results/<pod>.json` and `<pod>.log` (small, committed after download); GSM8K answers
  under `results/large/<pod>/acc/` (pod only); one checkpoint `results/large/<pod>/ckpt/<arm>.pt` ~17 GB, plus a .tmp while
  writing: ~34 GB peak of the pod's 250 GB disk. No δW is saved; nothing outlives the pod but the downloaded JSON and log.
- **Downloads.** The runner downloads `<pod>.json` and `<pod>.log` after each read, into the project, and checks sha256 on
  both sides before terminating.
- **Pod loss and resume.** The same command resumes from the last checkpoint (at most one read interval lost, ~25-50 steps);
  a lost pod's disk is gone, so a new pod restarts that arm from step 1 with the reads downloaded so far kept as the record.
- **Controls.** The control runs first and alone to its gate; N and F reproduce E69 at step 100 on the same card type.
- **Predictions.** Every rule run on edge cases and on W0 (above); the do-nothing state reads "not reached".
- **Code paths never run on a GPU.** `output_change` (first at step 5), the GSM8K read inside the training context at
  steps 10 and 25 (E69 ran the same `score` after step 100), the resume into `after_update`, and the gate. The control's
  first 25 steps run each of them except the resume before any other pod starts.

## Not in this run
No δW or states saved or uploaded (run 2 saves the 18 states); no seed 70; no N or F at the new rates; no GSM1K; no reading
on the decomposition (outside share, split, re-gating); no change to E69's files; no verdict on reuse; no learning-rate decay
or schedule search beyond the three recipes; no cluster smoke (the control pod is the rehearsal, its step-25 gate the stop).

## Appendix: commands per pod
Set-up on every pod, from the repo root (needs `WANDB_API_KEY`; `<sha>` = sha256 of `experiments/qwen_rl_readable/results/predictions.json` as committed):
```
uv sync --locked --no-default-groups --group dev --group gpu
export HF_HOME=/root/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 E120_PRED_SHA=<sha>
rm -f experiments/qwen_rl/results/large experiments/qwen_rl_readable/results/large
.venv/bin/python experiments/qwen_rl/fetch.py --root experiments/qwen_rl/results/large --only rollouts/H.jsonl
.venv/bin/python experiments/qwen_rl/fetch.py --root experiments/qwen_rl/results/large --only acc/W0_gsm8k
```
F only, before its run (E69's W0 rollouts, saves ~1.3 h of sampling; if the fetch fails, run.py samples them):
```
.venv/bin/python experiments/qwen_rl/fetch.py --lr-tag 1e-6w10 --root experiments/qwen_rl/results/large --only rollouts/F.tar
tar -xf experiments/qwen_rl/results/large/lr1e-6w10/rollouts/F.tar -C experiments/qwen_rl/results/large/rollouts
```
Run (`<pod>` = control, N, F, r3e-6 or r1e-5); the same command resumes from the last checkpoint after a crash:
```
set -o pipefail; .venv/bin/python experiments/qwen_rl_readable/run.py --pod <pod> 2>&1 | tee /proc/1/fd/1
```
Download `experiments/qwen_rl_readable/results/<pod>.json` and `<pod>.log` after each read; terminate once the final
JSON has `"done": true` (or `"stopped"`) and its sha256 matches on both sides. A `STOP:` line in the log is the job's end
even if the process then hangs at exit (a check's stop exits without `hard_exit`): the runner treats it as ended,
downloads, and does not wait on the process. After pods 1, 4 and 5 are home:
`~/commute-agents/fleet/cpu-run python experiments/qwen_rl_readable/run.py --score` writes `results/selection.json`.
