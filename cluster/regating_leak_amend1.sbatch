#!/bin/bash
# E18 amendment 1: why the shared pattern recovers less than the mean pattern on the word tasks (clip, per-window
# variation, or dead subcomponents). Plan: reports/plans/2026-10-21_plan_regating-leak.md -> Amendment 1.
#   cd ~/commute-agents/<role>
#   sbatch cluster/regating_leak_amend1.sbatch     # ~4 min on an L40: G1 + the E18 repeat gate (stops on a fail), then the arms
# No Pile stages (natural, adversary) are rerun. Before: `cluster/sync.sh up`; results/predictions_amend1.json filed;
# results/regating_leak.json (E18's results, the gate's reference) and E18's inputs on NFS (see regating_leak.sbatch).
#SBATCH --job-name=commute-regating-amend1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
for f in predictions_amend1.json regating_leak.json; do
  test -f "experiments/regating_leak/results/$f" || { echo "missing experiments/regating_leak/results/$f"; exit 1; }
done
.venv/bin/python experiments/regating_leak/run.py --amend1 "$@"
