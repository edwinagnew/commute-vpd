#!/bin/bash
# E283 reliance (P1.57): M's needed sets. Usage: sbatch --comment='E283 reliance: <job>' cluster/reliance.sbatch <tag> --stage ... [--world W] [--items ...] [--tasks ...]
#SBATCH --job-name=commute-reliance
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
TAG="$1"; shift 1
LOG=experiments/reliance/results/run_${TAG}.log
( .venv/bin/python experiments/reliance/run.py --tag "$TAG" "$@"; echo "=== FINISHED (exit $?) — results in $COMMUTE_REPO/experiments/reliance/results ===" ) 2>&1 | tee -a "$LOG"
