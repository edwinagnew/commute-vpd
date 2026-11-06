#!/bin/bash
# Upload E111's inputs to W&B (experiments/qwen_teacher_table/upload.py): "inputs" (E45's anchor-1 items and two TOFU
# files, 1.5 MB) and one artifact per δW E93 scored (qwen3-1.7b-e111-dW-<state>, 5.64 GB fp32 each; u2 2.82 GB bf16).
# A CPU batch job: the dev node's uplink is ~80 KB/s. Submit from the clone root, with WANDB_API_KEY in the submitting
# shell (sbatch exports it), naming what to upload; stage 1 needs only the first line:
#   sbatch cluster/qwen_teacher_table_upload.sbatch inputs R_s70
#   sbatch cluster/qwen_teacher_table_upload.sbatch N_s0 N_s70 F_s0 a1_s0 a1_s1 R_s0 a2_s42 a2_s142 u1 u2
# Each artifact's line in the log names it, its size, minutes and sha256 (E93's). Expected version v0 of each.
#SBATCH --job-name=commute-qwen_teacher_table_upload
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
trap 'echo "=== FINISHED (exit $?) ==="' EXIT
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
test -n "${WANDB_API_KEY:-}" || { echo "WANDB_API_KEY is not set"; exit 1; }
test "$#" -gt 0 || { echo "name what to upload: inputs and/or state names"; exit 1; }
.venv/bin/python experiments/qwen_teacher_table/upload.py upload "$@"
