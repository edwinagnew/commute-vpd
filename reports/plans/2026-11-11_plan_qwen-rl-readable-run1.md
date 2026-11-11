# Plan: which way of growing the RL update reaches 10 times the masked error while the pass rate holds? (E-TBD, cell P3.4 run 1)

## The question, in plain words
The saved reinforcement-learning (RL) updates on Qwen3-1.7B change the model's output by only 0.7 to 1.05 times the
decomposition's own masked error (0.0932 nats per token, E152 step 0), too small to read. Run 1 first reproduces E69's
GRPO loop at its chosen recipe (R, the shuffled-reward null N and the own-samples fine-tune F, seed 0, peak 1e-6 after 10
warm-up steps, 100 steps), then grows R's update three ways (more steps at 1e-6, peak 3e-6, peak 1e-5) and picks the
recipe that reaches the largest multiple of the masked error (10, 5 or 2 times) while its held-out GSM8K pass rate stays
within 0.03 of R at step 100 and above the unchanged model's. Run 2 then trains R, N and F on two seeds with that recipe.

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
No uploads, no δW saved; one checkpoint (~17 GB, pod disk only) at each read step for a resume.

## Pods (RunPod H100 80GB HBM3, in parallel; set-up and launch commands in the appendix below)
| pod | arm, rate | steps | reads | early stops (exit 2), from `results/predictions.json` |
|---|---|---|---|---|
| 1 control (first, alone) | R, 1e-6w10 | 250 | 25 50 100 150 200 250 | E69's step-1 and first-live checks; reward change by 20 > -0.05; step 20 \|δW\| in [0.1296, 0.1433]; step 100 \|δW\| in [0.333, 0.368], pass in [0.8587, 0.9027] |
| 2 N | N, 1e-6w10 | 100 | 100 | E69's checks; step 100 \|δW\| in [0.351, 0.388], pass in [0.848, 0.892] |
| 3 F | F, 1e-6w10 | 100 | 100 | E69's checks (no reward change); step 100 \|δW\| in [0.260, 0.288], pass in [0.7832, 0.8272] |
| 4 r3e-6 | R, 3e-6w10 | 200 | 25 50 100 150 200 | step-1 and first-live checks (KL band scaled to the rate), finite loss; drift stop |
| 5 r1e-5 | R, 1e-5w10 | 100 | 25 50 100 | as pod 4 |

**Stage gate.** Pods 2-5 start only after pod 1 logs `[check] step 20: dw_norm reproduces E69: ok` (about 1 h after its
start). Any stop in pod 1 before that: nothing else starts; tell the lead.

At each read (written to `results/<pod>.json` and logged as it lands, so the runner downloads as it goes): \|δW\|; the output
change, KL(W_t ‖ W0) per token over every position of the 128 maths rows at length 256, the quantity of E93's
`clean.W0.K` (`qwen_d3.maths_rows`, `vpd.kl_per_position(updated, W0)`, batched since the rows have no padding), and its
multiple of 0.0932; the greedy GSM8K pass rate (`qwen_eval.score`, strict_dropped, 1319 problems) with its paired gain over
W0. Expected at W0: 0, 0, 0.7817. Positive control for the output change: the control at step 100 should read about 0.0643
(E93 on E69's saved step-100 update; band 0.058-0.071, logged, not a stop: a miss goes to the lead before the sweep is read).
Blind to: the output change pools every maths position, so it cannot say where the change is; the pass rate is greedy only.

## Predictions (filed in `experiments/qwen_rl_readable/results/predictions.json`)
- **Reproduction.** The bands above (E69 seed 0 ±5% for \|δW\|, measured spread L40 vs H100 0.013 and seed 0 vs 70 0.004; pass
  ±0.022, 2 binomial se). At W0 every band misses; at E69's own values every band holds. *Lean: high*, same code, seed and card.
- **Selection** (`pick.py select`): bar = max(control's pass at 100 − 0.03, W0's 0.7817); a read is eligible at pass ≥ bar;
  its level is the largest of 10, 5, 2 its multiple reaches. Chosen: largest level, then fewer steps, higher pass, lower
  rate. Level ≥ 5: chosen; else "not reached" (RL could not get there at these recipes; run 2's design goes back to the lead);
  a recipe unfinished or no control read at 100: "incomplete". Run on edge cases before filing (no reads, a missing pass,
  all under the bar, ties in level/step/pass, exactly 10x at exactly the bar, W0 above the control's bar, only 2x, an
  unfinished recipe, no step-100 control read, W0 everywhere): each gave the intended outcome; W0 everywhere reads "not
  reached". Expected: the control alone about 0.7x at 100 and 1.5-2.5x at 250 (level 0 or 2). *Lean: medium* that 3e-6
  reaches 5x or 10x above the bar by step 200; the card's two rivals are the readings.
- **Drift stop** (`pick.py drift`, R pods): pass < 0.7817 while the output change is above the previous read's (0 before the
  first) ends the arm there; the read before is its last good one. Not expected on the control (pass ~0.88); possible on 1e-5.

## Cost (H100 $3.99/h; E69's measured H100 times: R step 56.4 s, N 76.9 s, F train 34.4 s from `lr1e-6w10/{R,N,F}.json`
steps; GSM8K read ~420 s from `lr1e-6w10/run.log` step 100 read to `[accuracy R gsm8k]`; set-up 40 min per fresh Qwen pod;
output change ~30 s and checkpoint ~60 s per read are not measured)
| pod | hours | $ |
|---|---|---|
| 1 control | 4.8 + 0.7 set-up = 5.5 | 21.7 |
| 2 N | 2.3 + 0.7 = 3.0 | 11.8 |
| 3 F (E69's F rollouts fetched; sampled instead: +1.3 h, +$5.2) | 1.1 + 0.7 = 1.8 | 7.2 |
| 4 r3e-6 (steps +20% for longer answers) | 4.5 + 0.7 = 5.2 | 20.6 |
| 5 r1e-5 (steps +20%) | 2.4 + 0.7 = 3.1 | 12.2 |
| total | wall clock ~6 h | ~73 |

The total is over CLAUDE.md's $60 gate for one run (each pod is under $60 and 6 h): the lead decides whether run 1 is
treated as gated. The cell brief allows $200 on RunPod without asking.

## Not in this run
No δW or states saved or uploaded (run 2 saves the 18 states); no seed 70; no N or F at the new rates; no GSM1K; no reading
on the decomposition (outside share, split, re-gating); no change to E69's files; no verdict on reuse; no learning-rate decay
or schedule search beyond the three recipes; no cluster smoke (the control pod is the rehearsal, its step-20 check the gate).

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
JSON has `"done": true` (or `"stopped"`) and its sha256 matches on both sides. After pods 1, 4 and 5 are home:
`~/commute-agents/fleet/cpu-run python experiments/qwen_rl_readable/run.py --score` writes `results/selection.json`.
