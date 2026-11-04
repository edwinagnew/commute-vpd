# Run brief: E78 (qwen_anchor1), anchor 1 retrained at E61's rate, δW in fp32

Plan `reports/plans/2026-11-03_plan_e78-anchor1.md`. Rules `results/predictions.json` (the run refuses without it). One cluster L40, `debug` QoS, $0, not gated. No WANDB key needed. Record job IDs. Route: if T has not started within 10 min, tell the lead (`.claude/CLAUDE.md` sends ≤ 30 min runs to RunPod then; T's 27 min is inside that but its 45-min `--time` may wait on backfill).

## Commands, from the clone root
```
sbatch cluster/qwen_anchor1.sbatch T                                  # about 27 min
sbatch --dependency=afterok:<T's id> cluster/qwen_anchor1.sbatch G    # about 10 min
```
Check first that `experiments/qwen_anchor1/results/large` is a symlink (`~/commute-agents/fleet/link-data <clone>` makes it; the run refuses otherwise). Log: `cluster/logs/commute-qwen_anchor1-<id>.out` and `experiments/qwen_anchor1/results/run.log`. Exit 0 done; 1 a stop (`results/stopped.json`, every stage refuses until the lead has read it and it is removed) or a refusal. Read `"done"` in the stage's JSON as well as the exit code. A T lost to the node starts again with the same command; its checks restart, saved files are overwritten. Never edit `results/predictions.json`.

## First minutes of T, against E61's run (all logged as `[check] ok|MISSED`)
| step | about | loss (E61, stop band) | |δW| (E61, stop band) | other |
|---|---|---|---|---|
| 1 | 25 s | 0.97258, [0.95473, 0.9937] | 0, exactly 0 | rate exactly 0; grad norm 56.95, note [42.8, 71.4] |
| 2 | 35 s | 1.10593, [1.08598, 1.13031] | 0.00336, [0.00269, 0.00403] | peak up to about 31.4 GiB, stop 40, note 33.5 |
| 3 | 1 min | 1.03678, note ≤ 1.0878 | 0.00978 | |
| 10 | 3 min | 0.34897, stop ≤ 0.95 | 0.12789 | maths KL 0.0354, held-out distillation KL 0.2538 |
| 20 | 5 min | 0.12831 | 0.30268 | 0.0896, 0.0993 |
| 50 | 13 min | 0.07555 | 0.46273 | 0.0931, 0.0536 |
| 100 | 27 min | 0.06260 | 0.48652 | 0.0959, 0.0473 |

A stop: tell the lead at once, do not resubmit. Steps 1-3 should match E61 to about 6 digits (E61's arm 0 repeat); a miss within the bands, or a growing gap, is reported, not a stop. Each `[save]` line takes about 20 s and writes 5.64 GB. Stall rule: a step takes about 12 s and a read step up to 40 s; no new log line for 3 min is a stall, tell the lead.

## G, in order
1. `[files]`: each of the 8 files 196 tensors, F32, norm within 1e-6 of the same step's |δW| in `train.json` (step 1 exactly 0). Then step 100 against E61's bf16: `one_minus_cos` ≤ 1e-3 and `norm_ratio` in [0.99, 1.01]. A miss stops before any generation. `rounding_alone: true` (≤ 5e-6, measured floor 1.39e-6) means the two runs agree to rounding; false with `holds: true` means the save path is right and the runs differ by more, so report it.
2. `[G] reproduction`: step 100 |δW|, loss, held, maths KL each within 1% of E61's; `step_gaps` is the measured repeat floor. A miss does not stop; report it to the lead before the files are used.
3. `[G] step 100`: `format_unprompted` about 0.926, `strict_dropped` about 0.621; rule `learned` needs ≥ 0.74 and ≥ 0.5603. `held_from_file` and `kl_maths_from_file` about 0.04725 and 0.09593. Peak GPU about 32 GiB.

## Outputs
- `~/commute-data/results-large/qwen_anchor1/dW/a1/step_{1,2,3,5,10,20,50,100}.safetensors`: 5.64 GB each, 45.1 GB. **Step 1 is all zeros by the schedule.**
- `results/large/answers/step_100_unprompted.jsonl`.
- `results/train.json`, `results/read.json`, `results/run.log`: small; commit them with the named paths, as run. Nothing else to download; it is all on NFS.

Report at each stage's end: step-100 numbers, `step_gaps`, the `against_e61` row, the rule, wall time, peaks.
