#!/bin/bash
# P1.27 held_model_size_matched. Usage: sbatch --comment='E### ...' cluster/held_model_size_matched.sbatch "1 2"   (stage list 1-9; "0" = stage 0 only, which every job runs first unless results/stage0.json says done)
#SBATCH --job-name=commute-held_model_size_matched
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS stages=$1" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/held_model_size_matched/results/predictions.json || { echo "no predictions"; exit 1; }
( .venv/bin/python experiments/held_model_size_matched/run.py $1 2>&1 | tee -a experiments/held_model_size_matched/results/run.log; exit ${PIPESTATUS[0]} )
rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/held_model_size_matched/results/ ==="
