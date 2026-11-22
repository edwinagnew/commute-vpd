#!/bin/bash
# P2.60 (E267), the lens's amendment 2: stage 0 then every item in one job. Usage:
#   PRED_SHA=$(git show HEAD:experiments/adl/results/amend2/predictions.json | sha256sum | cut -d' ' -f1) \
#     sbatch --comment=E267 --export=ALL cluster/adl_amend2.sbatch
# Then on CPU: ~/commute-agents/fleet/cpu-run python3 experiments/adl/amend2.py summary
#SBATCH --job-name=commute-adl-a2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
OUT=experiments/adl/results/amend2
( .venv/bin/python experiments/adl/amend2.py run; echo "=== FINISHED (exit $?) — results in $COMMUTE_REPO/$OUT ===" ) 2>&1 | tee -a "$OUT/run.log"
