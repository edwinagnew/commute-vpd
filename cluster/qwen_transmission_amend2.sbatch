#!/bin/bash
# E74 Amendment 2, residual-skip share and stochastic masks ($0): experiments/qwen_transmission/amend2.py. Plan: reports/plans/2026-11-03_plan_e74-transmission.md.
# Submit from the clone root: sbatch cluster/qwen_transmission_amend2.sbatch. Resumes from its read files; ~15 min expected.
#SBATCH --job-name=commute-qwen_transmission
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=01:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=$COMMUTE_REPO/experiments/qwen_maps:$COMMUTE_REPO/experiments/qwen_pass_test:$PYTHONPATH
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python -P experiments/qwen_transmission/amend2.py --decomp "$COMMUTE_QWEN_DIR"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
