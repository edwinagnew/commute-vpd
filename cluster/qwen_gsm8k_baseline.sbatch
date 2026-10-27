#!/bin/bash
# GSM8K baseline of the unchanged Qwen3-1.7B (Part 2 of reports/plans/2026-10-27_plan_qwen-port-and-baseline.md).
#   cd ~/commute-agents/<role>
#   sbatch cluster/qwen_gsm8k_baseline.sbatch    # ~12-18 min on an L40 incl. the ~3.4 GB model download to /ephemeral
# Stage 0 (grader on the gold solutions) runs before the model loads and stops the job on a fail.
#SBATCH --job-name=commute-qwen-gsm8k-baseline
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/qwen_gsm8k_baseline/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/qwen_gsm8k_baseline/run.py "$@"
