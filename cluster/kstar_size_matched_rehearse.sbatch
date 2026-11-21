#!/bin/bash
# E232 rehearsal: kstar_size_matched stage 0 + one scaled item, $0 on the cluster.
#   sbatch --comment='E232 rehearse: one scaled item' cluster/kstar_size_matched_rehearse.sbatch
#SBATCH --job-name=commute-e232-rehearse
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
( PRED_SHA=7fe94b66e8990a0695663b84f5de1b1b12aa1ded030f37a188dd407c549fabef /usr/bin/time -v .venv/bin/python experiments/kstar_size_matched/run.py --tag rehearse --out rehearse --items e54-cipher_own_s0_scaled_e54_own ) 2>&1 | tee -a experiments/kstar_size_matched/results/run_rehearse.log || rc=${PIPESTATUS[0]}
echo "=== FINISHED E232 rehearsal (exit $rc) — results in $COMMUTE_REPO/experiments/kstar_size_matched/results/ ==="
exit $rc
