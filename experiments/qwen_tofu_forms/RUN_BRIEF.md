# Run brief: E84 (qwen_tofu_forms), TOFU in five wordings

Plan `reports/plans/2026-11-04_plan_e84-rephrasings.md`. Rules `results/predictions.json`. Cluster L40, `debug`, $0, not gated. No WANDB key. Record job IDs. Never edit the predictions file.

## Command, from the clone root
```
sbatch cluster/qwen_tofu_forms.sbatch   # all four stages, 35 to 55 min; limit 1h50
```
RunPod only if `debug` has not started within 10 min. Log: `cluster/logs/commute-qwen_tofu_forms-<id>.out`, `experiments/qwen_tofu_forms/results/run.log`. The job's first lines print each card's used memory: over about 5 GB means someone holds it outside Slurm (problem #9 of qwen-decomp-lead), so report it. Exit 1 is a stop (`results/stopped.json`) or a refusal; tell the lead, do not resubmit. A lost job (timeout, node) is resubmitted with the same command: a stage whose result file says `"done": true` is skipped; a lost T starts again from step 0.

## 0, about 2 min
`[0] control, new-word recall, paraphrased` 0.167 and `never` 0.190, each within 0.02, or it stops (E81's readings 0.1673, 0.1895; a miss means the apparatus differs). Then `W0 on TOFU's own question` (expected 0.15 to 0.25) and `E81's first batch` (E81 read 4.782574); neither stops.

## R, 5 to 15 min
`[R] try 1` has each check's count. `first-try failed share` over 0.25 stops with "wordings: first-try drops over 25%": a filed outcome, hand back at once, the lead falls back to Haiku clerks. The file `results/large/forms_tries.jsonl` has every reply and what it missed. Up to 3 tries. Then `kept N rows, dropped M`; over 40 dropped stops. Ten `[R] sample` lines: a human reads these; quote two in the hand-back and say if any changed a fact. `results/forms.jsonl` about 1 MB.

## T, 20 to 30 min
| step | about | reads |
|---|---|---|
| start | | `[T] N sequences`, 4000 less 5 per dropped row; first batch loss |
| 1 | 1 min | loss within 1e-3 of that first batch, rate 0, dW 0 |
| 2 | 1 min | dW in [0.00208, 0.00312] (E81 0.00237); peak at most 44 GiB |
| 250 | 7 min | loss at most 0.95 of the first |
| 1250 | 30 min | a `[T]` line every 100 steps; E81 ended at loss 0.036 and dW 18.0 with one wording |

`[save]` lines take about 20 s, 5.64 GB each, 12 files. Stall rule: a step about 1.3 s, a save 20 s: no new log line for 5 min is a stall, tell the lead.

## G, about 6 min
`[files]`: 12 files, 196 tensors, F32, norm within 1e-6 of the step's record, step 1 exactly 0. `[G] W0`, then `[G] step 1250, 1000, 500, 200`: `para`, `trained`, `floor`, `key_para`, `key_trained`. Last line `[G] FILED RULE at step 1250`: answered, not answered, or training failed, with the three numbers. Report |dW| (expected 15 to 21) and that line.

## Outputs
`~/commute-data/results-large/qwen_tofu_forms/dW/a2_2e-5/step_{1,2,3,5,10,20,50,100,200,500,1000,1250}.safetensors` (68 GB); `results/large/answers/{W0,step_{200,500,1000,1250}}/{paraphrased,trained,never}.jsonl`; `results/large/forms_tries.jsonl`. `results/stage_0.json`, `stage_R.json`, `forms.jsonl`, `train.json`, `read.json`, `run.log` small and committed by named paths. All on NFS, nothing to download.
