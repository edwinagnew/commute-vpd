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

# On a pod (2026-11-04): stages T and G, because the cluster's free card is held outside Slurm (jobs 51561, 51562 died at model load)

Stages 0 and R are done and committed; their `"done": true` files arrive by `sync()`, so `run.py` starts at T. W&B carries files both ways: `experiments/qwen_tofu_forms/transfer.py` (docstring). Inputs already sent with `transfer.py send`. Everything above on bands, stalls and stops holds; times are for an H100 and may be shorter than an L40's.

**Pod.** One RunPod H100 80 GB, PCIe first, then SXM, then A100 80 GB; `gpu_type` passed explicitly per attempt; at most 2 failed starts, then hand back. Budget an H100 at $3.99/h. Container disk 250 GB, no network volume. Label `commute-qwen_tofu_forms-e84` through `start()`'s argument; never edit `remote-kernels.toml`. `instance=` on every machine-scoped call. `WANDB_API_KEY` in the pod's environment. Hand back past $15. Terminate if `start()` shows a different GPU or a price above $5/h.

**Set-up**, about 30 min, from the repo root, after `sync()`:
```
uv sync --locked --no-default-groups --group dev --group gpu
export HF_HOME=/root/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
for L in artifacts experiments/qwen_tofu_forms/results/large; do if [ -L $L ]; then rm $L; fi; done
mkdir -p artifacts /root/e84_large; ln -s /root/e84_large experiments/qwen_tofu_forms/results/large
.venv/bin/python experiments/qwen_tofu_forms/transfer.py receive
```
Other experiments' `results/large` links dangle and don't matter. `receive` prints 19 TOFU files and W0's three answer files. `results/large` must stay a symlink: `run.py` asserts it.

**Commands**, two background subprocesses started together, reads of at most 5 min (no `execute()` or `wait()` longer):
```
set -o pipefail; .venv/bin/python experiments/qwen_tofu_forms/run.py 2>&1 | tee -a experiments/qwen_tofu_forms/results/console.log /proc/1/fd/1
set -o pipefail; .venv/bin/python experiments/qwen_tofu_forms/transfer.py up 2>&1 | tee -a experiments/qwen_tofu_forms/results/upload.log /proc/1/fd/1
```
`run.py` exit 1 is a stop or a refusal: tell the lead, don't restart. `up` exits by itself when `read.json` says done, or on `stopped.json`.

**Home.** During the run, `download()` file by file to the same path `results/run.log`, `train.json`, `read.json` after each stage. At the end, in order:
1. `up` printed `[up] done, 13 files`, then `.venv/bin/python experiments/qwen_tofu_forms/transfer.py verify` prints `[verify] all home on W&B`.
2. `download()` `results/read.json`, `train.json`, `run.log`, `SHA256SUMS` to the same paths; sha256 of `read.json` on both sides matches.
3. `terminate()` at once, before any local file move.
4. Commit those four by named paths, push, see the push land.
5. From the clone root: `sbatch cluster/qwen_tofu_forms_fetch.sbatch`, CPU, about 68 GB, about 30 min, every file checked against `SHA256SUMS`. Wait with a background `wait-job`. Last log line `[down] done`.

# Amendment 1 (2026-11-04): stages T and G again at 10 epochs

E84 missed its line (para 0.5316 against 0.6021, trained 0.9927, floor 0.2115). The plan's one amendment: the same T and G with 2500 steps (10 epochs by E81's rule, 10 * 4000 // 16) instead of 1250; everything else as E84 (3935 sequences, peak rate 2e-5, warm-up 250, batch 16, micro 8, seed 42, fp32). Rules: `results/predictions_amend1.json`. A miss goes to the user; no third.

**Command**, from the clone root: `sbatch cluster/qwen_tofu_forms_amend1.sbatch` (T then G; `--stage G` alone). Exit 1 is a stop (`results/stopped_amend1.json`) or a refusal: tell the lead, don't restart. Bands and stall rule as above: step 1 loss within 1e-3 of the first batch, step 2 |dW| in [0.00208, 0.00312], step 250 loss at most 0.95 of the first, peak at most 44 GiB; a step about 1.3 s, a save 20 s, no log line for 5 min is a stall. T about 55 min, G about 5 min (`--time` 02:00:00).

**Last lines.** `[G] FILED RULE at step 2500`: label with para, trained, floor (expected para above 0.532, trained about 1.0, floor about 0.22); report |dW| at 2500 (expected 18 to 30, no rule). Not rerun: W0 (graded from `results/large/answers/W0`).

**Outputs.** `~/commute-data/results-large/qwen_tofu_forms/dW/a2_2e-5_e10/step_{1,2,3,5,10,20,50,100,200,500,1000,1250,2000,2500}.safetensors` (14 x 5.64 GB); `results/large/answers_e10/step_{2500,1250}/{paraphrased,trained,never}.jsonl`; `results/amend1_train.json`, `amend1_read.json`, `run_amend1.log`, small, committed by named paths. All on NFS.

A job on the held card dies within seconds at model load (jobs 51561, 51562): if so, tell the lead, don't retry in a loop.
