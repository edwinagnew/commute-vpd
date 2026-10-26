#!/bin/bash
# Complement revert (E20), amendment 1: R split into the dead block and Delta'. Plan:
# reports/2026-10-21_plan_cancellation-complement.md -> Amendment 1.
#   cd ~/commute-agents/<role>
#   sbatch cluster/cancellation_complement_amend1.sbatch    # ~10 min: W0 (gates first) and eight states, 128 rows
# Before that: `cluster/sync.sh up`, results/predictions_amend1.json filed, and E20's results/complement.json present
# (gate G2 reads it). Inputs on NFS as E20: experiments/reading_a_coverage/results/large/<task>/{pullback_s0/
# state_step10.pt, dW_ft_step10_s0.pt}, experiments/transport_test/results/large/{alive.pt, windows.pt}, artifacts/.
#SBATCH --job-name=commute-complement-a1
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
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=amend1" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
R=experiments/cancellation_complement/results
test -f $R/predictions_amend1.json || { echo "file the predictions first (run.py --predictions amend1)"; exit 1; }
test -f $R/complement.json || { echo "E20's complement.json missing (gate G2 reads it)"; exit 1; }
.venv/bin/python experiments/cancellation_complement/run.py amend1
