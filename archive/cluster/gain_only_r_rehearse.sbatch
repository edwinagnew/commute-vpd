#!/bin/bash
# E236 rehearsal: stage 0 + reproduce of e58/cipher_s0 + case_label/twin_s0, full read sizes. Resubmit until it ends clean (every reader resumes from its own file).
#SBATCH --job-name=commute-gain-only-r-rehearse
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
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/gain_only_r/results/predictions.json || { echo "file the predictions first"; exit 1; }
LOG=experiments/gain_only_r/results/run_rehearse.log
( .venv/bin/python experiments/gain_only_r/run.py --tag rehearse --out rehearse --reproduce --items e58/cipher_s0 case_label/twin_s0; echo "=== FINISHED (exit $?) — results in $COMMUTE_REPO/experiments/gain_only_r/results/rehearse/ and stage0_rehearse.json ===" ) 2>&1 | tee -a "$LOG"
