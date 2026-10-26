#!/bin/bash
# E18 amendment 2: a random null for the whole shared change (the shift shuffled over each matrix's alive
# subcomponents, 20 draws, clipped and not) and the no-clip shift by bins of its size.
# Plan: reports/plans/2026-10-21_plan_regating-leak.md -> Amendment 2.
#   cd ~/commute-agents/<role>
#   sbatch cluster/regating_leak_amend2.sbatch     # ~2-4 min on an L40: G1 + the exact repeat of amendment 1 (stops on a fail), then the arms
# Before: `cluster/sync.sh up`; results/predictions_amend2.json filed; results/regating_leak_amend1.json (the gate's
# reference) and E18's inputs on NFS (see regating_leak.sbatch).
#SBATCH --job-name=commute-regating-amend2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
for f in predictions_amend2.json regating_leak_amend1.json; do
  test -f "experiments/regating_leak/results/$f" || { echo "missing experiments/regating_leak/results/$f"; exit 1; }
done
.venv/bin/python experiments/regating_leak/run.py --amend2 "$@"
