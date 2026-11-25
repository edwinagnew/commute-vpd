#!/bin/bash
# E79 ($0, ~6 min): E76's saved states read for agreement, entropy, unseen text and own masks. g per row on the node's disk,
# recomputed if absent. Submit from the clone root: sbatch cluster/qwen_masked_read.sbatch
#SBATCH --job-name=commute-qwen_masked_read
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
set +e
.venv/bin/python experiments/qwen_masked_read/run.py --decomp "$COMMUTE_QWEN_DIR" --out "/ephemeral/$USER/e79" "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
