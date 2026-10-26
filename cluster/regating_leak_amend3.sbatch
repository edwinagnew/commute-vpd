#!/bin/bash
# E18 amendment 3: arms the leading hypothesis H can fail. The state's own trigger values with the fixed change removed,
# another window's values (5 derangements, per-window gains), the change's turned-up and turned-down halves against
# sign-kept shuffles (20 draws each), and the change on Pile rows.
# Plan: reports/plans/2026-10-21_plan_regating-leak.md -> Amendment 3.
#   cd ~/commute-agents/<role>
#   sbatch cluster/regating_leak_amend3.sbatch     # ~3-5 min on an L40: G1 + the exact repeat of amendment 2 (stops on a fail), then the arms
# Before: `cluster/sync.sh up`; results/predictions_amend3.json filed; results/regating_leak_amend2.json (the gate's
# reference) and E18's inputs on NFS (see regating_leak.sbatch); Pile val rows 0..63 cached (E18's adversary stage).
#SBATCH --job-name=commute-regating-amend3
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
for f in predictions_amend3.json regating_leak_amend2.json; do
  test -f "experiments/regating_leak/results/$f" || { echo "missing experiments/regating_leak/results/$f"; exit 1; }
done
.venv/bin/python experiments/regating_leak/run.py --amend3 "$@"
