#!/bin/bash
# Fresh-number check of the unchanged Qwen3-1.7B on GSM8K (reports/plans/2026-10-27_plan_qwen-fresh-numbers.md).
#   cd ~/commute-agents/<role>
#   sbatch cluster/qwen_fresh_numbers.sbatch    # ~22 min on an L40: 32 batches of 128 at ~37 s, plus model and data
# Stage 0 (grader on the gold solutions of all three sets) runs before the model loads; stage 1 (GSM8K test, the
# reference) must reproduce the baseline's 0.756 or its responses; either failure stops the job.
#SBATCH --job-name=commute-qwen-fresh-numbers
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
test -f experiments/qwen_fresh_numbers/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
rc=0; .venv/bin/python experiments/qwen_fresh_numbers/run.py "$@" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/qwen_fresh_numbers/results/fresh.json and $COMMUTE_REPO/experiments/qwen_fresh_numbers/results/large/ ==="
exit $rc
