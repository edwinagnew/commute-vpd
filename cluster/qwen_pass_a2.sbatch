#!/bin/bash
# E68 Amendment 2, the masks opened in steps ($0, one L40, debug QoS): experiments/qwen_pass_test/amend1.py.
# Plan: reports/plans/2026-11-03_plan_e68-d1-pass-test.md. Past --wall-min it exits 0: resubmit, it resumes.
# Outputs: experiments/qwen_pass_test/results/amend1/ (open.json, run.log, one JSON per read). --mem as E68.
#SBATCH --job-name=commute-qwen_pass_a2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# E44 Amendment 2's reader and stage S; -P keeps this script's folder off sys.path, so amend2.py's `run` is qwen_maps/run.py
export PYTHONPATH=$COMMUTE_REPO/experiments/qwen_maps:$COMMUTE_REPO/experiments/qwen_pass_test:$PYTHONPATH
nvidia-smi --query-gpu=name,memory.total --format=csv
df -h /ephemeral
set +e
.venv/bin/python -P experiments/qwen_pass_test/amend1.py --decomp "$COMMUTE_QWEN_DIR" --wall-min 26 --open
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
