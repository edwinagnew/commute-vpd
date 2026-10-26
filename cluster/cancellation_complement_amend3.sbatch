#!/bin/bash
# Complement revert (E20), amendment 3: the adversary at budgets of 20/40/80/160 sign steps, step size 2/steps. Plan:
# reports/2026-10-21_plan_cancellation-complement.md -> Amendment 3.
#   cd ~/commute-agents/<role>
#   sbatch --export=ALL,SHARD=0 cluster/cancellation_complement_amend3.sbatch   # W0 (control, gate C3) FIRST, alone, ~16 min
#   then SHARD=1, 2, 3 (each refuses to start until shard 0's file shows C3 passed); ~16, ~16, ~9 min
# Before: `cluster/sync.sh up`, results/predictions_amend3.json filed, amendment 1's results/complement_amend1.json
# present. Inputs on NFS as E20: E11's results/large/<task>/, experiments/transport_test/results/large/, artifacts/.
#SBATCH --job-name=commute-complement-a3
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
: "${SHARD:?set SHARD=0..3 via --export}"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=amend3_budget shard=$SHARD" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
R=experiments/cancellation_complement/results
test -f $R/predictions_amend3.json || { echo "file the predictions first (run.py --predictions amend3_budget)"; exit 1; }
test -f $R/complement_amend1.json || { echo "amendment 1's complement_amend1.json missing"; exit 1; }
.venv/bin/python experiments/cancellation_complement/run.py amend3_budget --shard "$SHARD"
