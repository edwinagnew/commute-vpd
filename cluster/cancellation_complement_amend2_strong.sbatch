#!/bin/bash
# Complement revert (E20), amendment 2, stage amend2_strong. Plan:
# reports/2026-10-21_plan_cancellation-complement.md -> Amendment 2.
#   cd ~/commute-agents/<role>
#   sbatch cluster/cancellation_complement_amend2_strong.sbatch    # ~12-14 min: the stronger adversary at W0 (control C2 first), then r1 t534; 190 sign steps per model
# Before that: `cluster/sync.sh up`, results/predictions_amend2.json filed, and amendment 1's
# results/complement_amend1.json and results/large/sources/*.pt present (both stages read them). Inputs on NFS as E20:
# experiments/reading_a_coverage/results/large/<task>/pullback_s0/state_step10.pt (and dW_ft_step10_s0.pt for the
# splits), experiments/transport_test/results/large/{alive.pt, windows.pt}, artifacts/.
#SBATCH --job-name=commute-complement-a2-strong
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
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=amend2_strong" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
R=experiments/cancellation_complement/results
test -f $R/predictions_amend2.json || { echo "file the predictions first (run.py --predictions amend2_fixed)"; exit 1; }
test -f $R/complement_amend1.json || { echo "amendment 1's complement_amend1.json missing"; exit 1; }
test -f $R/large/sources/W0.pt || { echo "amendment 1's saved sources missing (results/large/sources/)"; exit 1; }
.venv/bin/python experiments/cancellation_complement/run.py amend2_strong
