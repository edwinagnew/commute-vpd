#!/bin/bash
# E174 (P3.4/P3.8 library share), run 2, arms known,invented on the cluster (normal QoS, L40). Submit from the clone root:
#   sbatch --comment='E174 stage: known-invented' cluster/qwen_library_share_run2_known-invented.sbatch
#SBATCH --job-name=commute-E174-known-invented
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export E120_PRED_SHA=d55fa0f502d937cafd6c89ce497afb727ba91e8dd9bd441c523b9737467aec2d
LOG=experiments/qwen_library_share/results/run2_known-invented_pod.log
.venv/bin/python experiments/qwen_library_share/run.py --run 2 --arms known,invented \
  --decomp "$HOME/commute-data/artifacts/qwen3-1.7b-p-9707d6ff-torch" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
echo "=== FINISHED (exit $status) — results in $COMMUTE_REPO/experiments/qwen_library_share/results/run2/ and $COMMUTE_REPO/$LOG ===" | tee -a "$LOG"
exit $status
