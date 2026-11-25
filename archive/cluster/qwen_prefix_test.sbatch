#!/bin/bash
# Prefix test of Qwen3-1.7B-Base on GSM8K (reports/plans/2026-10-28_plan_qwen-prefix-test.md).
#   cd ~/commute-agents/<role>
#   sbatch cluster/qwen_prefix_test.sbatch    # ~10 min on an L40: 3.4 GB base-model download into $HF_HOME
#                                             # (/ephemeral, cluster/env.sh), then 40 batches of 128 short completions
# Stage 1 (copy control: the model copies each question it was just shown) runs first and alone; below 0.90 the job
# stops. Each stage has a time budget (120, 300, 420, 300 s) and records how many items it reached.
#SBATCH --job-name=commute-qwen-prefix-test
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
test -f experiments/qwen_prefix_test/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
rc=0; .venv/bin/python experiments/qwen_prefix_test/run.py "$@" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/qwen_prefix_test/results/prefix.json and $COMMUTE_REPO/experiments/qwen_prefix_test/results/large/ ==="
exit $rc
