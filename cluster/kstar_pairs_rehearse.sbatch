#!/bin/bash
# E221 rehearsal: kstar_pairs stage 0 + two items, $0 on the cluster.
#   sbatch --comment='E221 rehearsal' cluster/kstar_pairs_rehearse.sbatch
#SBATCH --job-name=commute-e221-rehearse
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
( PRED_SHA=dfbc9c28f0658147cbd7ec2db8c42a83b49dee762c83a3fff9ec4817940b5245 .venv/bin/python experiments/kstar_pairs/run.py --tag rehearse --out rehearse --items twin_s0 e58/cipher_s0 ) 2>&1 | tee -a experiments/kstar_pairs/results/run_rehearse.log || rc=${PIPESTATUS[0]}
echo "=== FINISHED E221 rehearsal (exit $rc) — results in $COMMUTE_REPO/experiments/kstar_pairs/results/ ==="
exit $rc
