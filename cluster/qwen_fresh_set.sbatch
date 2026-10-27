#!/bin/bash
# Fresh-set check of the unchanged Qwen3-1.7B: GSM8K test against GSM1k and 2026 GSM-Symbolic variants
# (reports/plans/2026-10-28_plan_qwen-fresh-set.md).
#   cd ~/commute-agents/<role>
#   sbatch cluster/qwen_fresh_set.sbatch    # ~22 min on an L40: 29 batches of 128 at ~37-41 s, plus model and data
# Stage 0 (grader on every set's gold answers) runs before the model loads; stage 1 (GSM8K test, the reference) must
# reproduce the baseline's 0.756 or its responses; either failure stops the job. Each stage has a time budget
# (480, 450, 360 s) and records how many items it reached.
#SBATCH --job-name=commute-qwen-fresh-set
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
test -f experiments/qwen_fresh_set/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
rc=0; .venv/bin/python experiments/qwen_fresh_set/run.py "$@" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/qwen_fresh_set/results/fresh_set.json and $COMMUTE_REPO/experiments/qwen_fresh_set/results/large/{gsm8k,gsm1k,mgs}.jsonl ==="
exit $rc
