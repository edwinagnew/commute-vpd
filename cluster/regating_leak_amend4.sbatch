#!/bin/bash
# E18 amendment 4: does the remainder carry window-specific information? Each window gets another window's remainder.
# Plan: reports/plans/2026-10-21_plan_regating-leak.md -> Amendment 4.
#   cd ~/commute-agents/<role>
#   sbatch cluster/regating_leak_amend4.sbatch     # ~3-5 min on an L40: G1 + amendment 3's gate and remainder (stops on a fail), then the arms
# Before: `cluster/sync.sh up`; results/predictions_amend4.json filed; results/regating_leak_amend{2,3}.json (the gate's
# reference) and E18's inputs on NFS (see regating_leak.sbatch); Pile val rows 0..63 cached (E18's adversary stage).
#SBATCH --job-name=commute-regating-amend4
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
for f in predictions_amend4.json regating_leak_amend2.json regating_leak_amend3.json; do
  test -f "experiments/regating_leak/results/$f" || { echo "missing experiments/regating_leak/results/$f"; exit 1; }
done
.venv/bin/python experiments/regating_leak/run.py --amend4 "$@"
