# Run brief: E44's input remade, E40's no-warm-up δW on the cluster ($0, MATS `debug`, one L40), then to W&B

Why: E44 reads E40's δW at steps 1, 10, 100 from RunPod volume `ydavm7ct4x` (US-MO-1), where 6 of 6 pod starts failed.
The user chose (2026-10-29) to remake them on the cluster, upload them to W&B and run E44 on any H100. Plan:
`reports/plans/2026-10-28_plan_qwen-q0q1.md`, "Amendment record (2026-10-29)" (what differs from E40, and why C reruns).
Code: `regen.py` (beside this file), `src/commute/qwen_train.py` (the step), `src/commute/qwen_e40_files.py` (upload, fetch).

**Before the first job**, in the lead's clone after `git pull`: `ls ~/commute-data/results-large/qwen_sft/` (the job
writes `nowarmup_regen/` there; nothing else is touched). Submit from the clone root, one job at a time. Record job IDs.

**Exit codes (job 1).** 0 done and `dW/SHA256SUMS` written. **2 = STOP**: step 1's loss or KL to start is off E40's by
more than 1% (`results/regen.json` `"stopped"`): report to the lead, no further job. 3 = paused at a checkpoint (only
with `--pause`): resubmit the same command.

| # | command | L40 time | first values to see in `cluster/logs/commute-qwen_e40_regen-<id>.out` | stop rule (code) | output |
|---|---|---|---|---|---|
| 1 | `sbatch --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True cluster/qwen_e40_regen.sbatch` | ~13-16 min (load ~1, 100 steps at ~5 s, 3 KL reads ~30 s each, saves ~1, sha256 ~2) | within 2 min: `step 1: loss ~2.4070` (E40 read 2.4102 on its H100), KL to start `"kl": ~0.1068, "e40": 0.10621, "rel": ~+0.005, "within_tol": true`; step 2 loss ~1.52; step 10 `"e40": 0.52040`; step 100 `"e40": 0.66369`; `[save]` lines for `step_1`, `step_10`, `step_100` (2.82 GB) and `step_100_fp32` (5.64 GB); last: SHA256SUMS listing and peak GPU (~40 GB) | step 1's loss or KL off E40 by > 1% → exit 2 | `~/commute-data/results-large/qwen_sft/nowarmup_regen/dW/`, `experiments/qwen_maps/results/regen.json` |
| — | **read `results/regen.json` `kl`**: steps 10 and 100 `within_tol` | | both true → job 2. Either false → **stop, report the three `rel` values to the lead** (E44's C on the pod would stop on the same 1% rule; the lead decides whether to upload anyway) | | |
| 2 | `sbatch cluster/qwen_e40_upload.sbatch` (WANDB_API_KEY in the shell; if missing, name it and stop) | ~5-15 min (sha256 check ~1, 14 GB upload) | `[upload] edwinagnew-mats/commute/qwen3-1.7b-e40-nowarmup-dW:v0 in N min; sha256 {...}` | a file not matching SHA256SUMS → AssertionError, report | the W&B artifact |
| 3 | the fetch a pod will make, as a check: `~/commute-agents/fleet/cpu-run -c 4 --mem 4G -t 00:30:00 -- env PYTHONPATH=src .venv/bin/python -m commute.qwen_e40_files fetch ~/commute-data/scratch/e40_fetch_check` | ~5-10 min | `[fetch] dW/step_1.safetensors: 2820 MB in N s` per file, exit 0 (the fetch ends with `sha256sum -c`) | any exception → report | then `rm -r ~/commute-data/scratch/e40_fetch_check` |

**If job 1 hits the 30-min limit** (no `[regen] done` line): resubmit job 1 with `--pause 50` appended. It starts
afresh (the old `dW/` is deleted, so no file mixes two trajectories), checkpoints after step 50 (~17 GB in
`nowarmup_regen/ckpt.pt`, deleted at the end) and exits 3; resubmit the same command to finish.

**Reporting.** After each job: job ID, exit code, wall time, peak GPU. Job 1: loss at step 1, the three KL-to-start
`rel` values, the four file sizes and SHA256SUMS. Job 2: the artifact's qualified name and version. Job 3: exit code.
Commit `experiments/qwen_maps/results/regen.json` (never the δW). Any STOP or surprise goes to the lead first.
Expected total: 1 GPU `debug` job + 2 CPU jobs, under 45 min, $0, 14.1 GB on `commute-data`.
