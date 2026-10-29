# Run brief: E44's input remade, E40's no-warm-up δW on the cluster ($0, MATS `debug`, one L40), then to W&B

Why: E44 reads E40's δW at steps 1, 10, 100 from RunPod volume `ydavm7ct4x` (US-MO-1), where 6 of 6 pod starts failed.
The user chose (2026-10-29) to remake them on the cluster, upload them to W&B and run E44 on any H100. Plan:
`reports/plans/2026-10-28_plan_qwen-q0q1.md`, "Amendment record (2026-10-29)"; review
`reports/reviews/2026-10-29_review_e40-regen-code.md` (its four findings fixed in code). Code: `regen.py` (beside this
file), `src/commute/qwen_train.py` (the step), `src/commute/qwen_e40_files.py` (upload, fetch).

**Before the first job**, in the lead's clone after `git pull`: `ls ~/commute-data/results-large/qwen_sft/` (the job
writes `nowarmup_regen/` there). If an earlier job left `nowarmup_regen/dW/SHA256SUMS`, regen refuses a fresh start:
ask the lead before adding `--force`. Submit from the clone root, one job at a time. Record job IDs.

**Exit codes (job 1).** 0: every check held, `dW/SHA256SUMS` written and copied to
`experiments/qwen_maps/results/regen_SHA256SUMS`. **2 = STOP**: a check missed (`STOP:` line, `results/regen.json`
`"stopped"`); no sums are written and the upload refuses; report to the lead, no further job. 3 = paused at a
checkpoint (only with `--pause`): resubmit the same command.

**Checks in code (each stops the step it misses).** (a) E44's stage C, computed here as C computes it: KL to start from
the bf16 files at 1 and 10 within 1% of E40's 0.10621 and 0.52040 (then again within 1% + 3 × gap once step 100 gives
the gap), from the fp32 weights at 100 within 1% of 0.66369. (b) E40's record (`qwen_sft/results/F.json`): the training
loss at every step, and |δW| of each of the 196 matrices at 1, 10, 100, within 1% at steps 1-10 (E44's tol, ~8× the
0.13% card gap at step 1) and 5% at 11-100 (E45's filed L40 training-curve tolerance).

| # | command | L40 time | first values to see in `cluster/logs/commute-qwen_e40_regen-<id>.out` | output |
|---|---|---|---|---|
| 1 | `sbatch --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True cluster/qwen_e40_regen.sbatch` | ~15-18 min (load ~1, 100 steps at ~5 s, 6 KL reads ~30 s each, saves ~1, sha256 ~2) | within 2 min: `step 1: loss ~2.4070` (E40 2.4102), KL (fp32 weights) ~0.1068, then `KL to start {"1": ~0.107}` (bf16 file; E40 0.10621, limit 1%) and `|δW| worst rel` < 1e-2; step 2 loss ~1.52; step 10 KL near 0.520; step 100 near 0.664; `[save]` lines for `step_1`, `step_10`, `step_100` (2.82 GB) and `step_100_fp32` (5.64 GB); last: `done, every check held`, SHA256SUMS listing, peak GPU (~40-45 GB) | `~/commute-data/results-large/qwen_sft/nowarmup_regen/dW/`, `results/regen.json`, `results/regen_SHA256SUMS` |
| — | **commit and push** `experiments/qwen_maps/results/regen.json` and `regen_SHA256SUMS` before job 2 (the pod's fetch checks against the committed sums) | | | |
| 2 | `sbatch cluster/qwen_e40_upload.sbatch` (WANDB_API_KEY in the shell; if missing, name it and stop) | ~5-15 min | `[upload] edwinagnew-mats/commute/qwen3-1.7b-e40-nowarmup-dW:vN in N min` — **report vN**; it refuses (AssertionError) unless regen.json has `done` and `pass` and the files match the committed sums | the W&B artifact |
| 3 | fetch check: `~/commute-agents/fleet/cpu-run -c 4 --mem 4G -t 00:30:00 -- env PYTHONPATH=src .venv/bin/python -m commute.qwen_e40_files fetch ~/commute-data/scratch/e40_fetch_check vN` | ~5-10 min | `[fetch] dW/step_1.safetensors: 2820 MB in N s` per file, exit 0 (ends with the committed-sums check) | then `rm -r ~/commute-data/scratch/e40_fetch_check` |

**If job 1 hits the 30-min limit** (no `done` line): resubmit job 1 with `--pause 50` appended. With no checkpoint and
no SHA256SUMS it starts afresh (the unfinished `dW/` is deleted, so no file mixes two trajectories), checkpoints after
step 50 (~17 GB in `nowarmup_regen/ckpt.pt`, deleted at the end) and exits 3; resubmit the same command, which resumes
from the checkpoint's own step.

**Reporting.** After each job: job ID, exit code, wall time, peak GPU. Job 1: step-1 loss, the KL-to-start values and
gap, the largest loss and |δW| relative gaps, file sizes and SHA256SUMS. Job 2: the artifact version vN (the pod's
fetch names it; put it in `RUN_BRIEF.md` section 4). Job 3: exit code. Never commit the δW. Any STOP or surprise goes
to the lead first. Expected total: 1 GPU `debug` job + 2 CPU jobs, under 50 min, $0, 14.1 GB on `commute-data`.
