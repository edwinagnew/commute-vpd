#!/bin/bash
# E57 (Q3) Stage A on one cluster L40, debug QoS ($0): experiments/qwen_reading/run.py --stage A. Plan:
# reports/plans/2026-10-30_plan_q3-reading-the-update.md, sections 6.1 and 7. Submit from the clone root; not
# submitted by the code writer:
#   sbatch cluster/qwen_reading_A.sbatch
# The controls run first and alone (W0 as FT, E40 step 10's first 16 rows against Amendment 2, do-nothing, the GPU
# rule); a miss writes experiments/qwen_reading/results/stopped.json and exits 1. Then the split-half floor, the gate
# count check, the plain and prompted answer sets, and row 4 (a)-(b) (records E45's sha256 in results/dW_SHA256SUMS).
# Expected ~25 min (controls ~6 min). GPU ~42.5 GiB in the cells (Amendment 2's 42.3 at 256 tokens). Host: Σ 11 GB
# + two pinned weight sets 11 GB + δW, over E44 Amendment 2's 56.6 GB at S: 96G (plan section 8).
#SBATCH --job-name=commute-qwen_reading_A
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_reading/run.py --stage A --decomp "$COMMUTE_QWEN_DIR" "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
