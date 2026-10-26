#!/bin/bash
# E30 amendment 1: the raised and the lowered half of the fixed change, each applied everywhere (emoticon SGD 0.01).
# Plan: reports/2026-10-22_plan_commute-second-line.md -> "Amendment 1 to E30".
#   cd ~/commute-agents/<role>
#   sbatch cluster/gamma_shift_map_amend1.sbatch   # ~18 min on a 4090 (8 Pile readings of ~2 min); C1 first, stops on a fail
# Before: `cluster/sync.sh up`; results/predictions_amend1.json filed; results/gamma_shift_map.json (E30, committed);
# experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt} on NFS. Same card for the whole run:
# never resume results/gamma_shift_map_amend1.json from another card type (E30 attempt 2 missed C1 that way).
#SBATCH --job-name=commute-gsm-amend1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
for f in experiments/gamma_shift_map/results/predictions_amend1.json experiments/gamma_shift_map/results/gamma_shift_map.json \
         experiments/transport_test/results/windows.json experiments/transport_test/results/large/alive.pt \
         experiments/transport_test/results/large/windows.pt; do
  test -f "$f" || { echo "missing $f"; exit 1; }
done
.venv/bin/python experiments/gamma_shift_map/halves.py "$@"
