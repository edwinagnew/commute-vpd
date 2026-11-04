# Run brief: E81 (qwen_tofu), anchor 2, TOFU

Plan `reports/plans/2026-11-04_plan_e81-tofu.md`. Rules `results/predictions.json`. Cluster L40s, `debug`, $0, not gated. No WANDB key. Record job IDs. Never edit predictions.json.

## Commands, from the clone root
```
sbatch --time=00:30:00 cluster/qwen_tofu.sbatch --stage 0                                  # about 10 min
sbatch cluster/qwen_tofu.sbatch --stage T --rate 1e-5                                      # about 70 min; only after 0 passes
sbatch cluster/qwen_tofu.sbatch --stage T --rate 2e-5                                      # same, in parallel
sbatch --time=00:40:00 --dependency=afterok:<T's id> cluster/qwen_tofu.sbatch --stage G --rate <rate>   # about 15 min
```
Log: `cluster/logs/commute-qwen_tofu-<id>.out`, `experiments/qwen_tofu/results/run.log`. The job's first lines print each card's used memory: over about 5 GB means someone holds it outside Slurm (problem #9 of qwen-decomp-lead), so report it. Exit 1 is a stop (`results/stopped.json`) or a refusal; tell the lead, do not resubmit.

## Stage 0
In order: `[0] first batch loss`, `[0] real facts` (about 0.8; below 0.5 stops), `[0] paraphrased questions`, `[0] original questions` (each about 1/6), recalls. Exit 1 with "not at chance" is a filed outcome: hand back at once.

## T
| step | about | reads |
|---|---|---|
| 1 | 1 min | loss within 1e-3 of stage 0's `first_batch_loss`, rate 0, dW 0 |
| 2 | 1 min | dW in [0.00104, 0.00156] at 1e-5, [0.00208, 0.00312] at 2e-5; peak at most 44 GiB |
| 250 | 15 min | loss at most 0.95 of the first |
| 1250 | 70 min | `choice` every 100 steps in the log, from about 1/6 up |

`[save]` lines take about 20 s, 5.64 GB each, 12 files per rate. Stall rule: a step about 2 s, a `choice` read about 1 min, a save 20 s: no new log line for 5 min is a stall (the log prints only at read steps), tell the lead. A lost T starts again with the same command. A timeout at 1h50 is a hand-back: the run has no checkpoint.

## G
`[files]`: 12 files, 196 tensors, F32, norm within 1e-6 of the step's record, step 1 exactly 0. Then `[G] step <t>`: `paraphrased.share`, `real.share`, three recalls, `answered`.

## Outputs
`~/commute-data/results-large/qwen_tofu/dW/a2_<rate>/step_{1,2,3,5,10,20,50,100,200,500,1000,1250}.safetensors`; `results/large/answers/`; `results/stage0.json`, `train_<rate>.json`, `read_<rate>.json`, `run.log` small and committed by named paths. All on NFS, nothing to download.
