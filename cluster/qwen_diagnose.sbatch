#!/bin/bash
# E73, clean-model diagnostics ($0): experiments/qwen_diagnose/run.py. Plan: reports/plans/2026-11-03_plan_e73-diagnose.md.
# Submit from the clone root: sbatch cluster/qwen_diagnose.sbatch. Resumes from its read files.
#SBATCH --job-name=commute-qwen_diagnose
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=$COMMUTE_REPO/experiments/qwen_maps:$COMMUTE_REPO/experiments/qwen_pass_test:$PYTHONPATH
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/python -P experiments/qwen_diagnose/run.py
echo "=== FINISHED ==="
