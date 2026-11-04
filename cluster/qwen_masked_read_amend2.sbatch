#!/bin/bash
# E79 Amendment 2 ($0, ~3 min): does A's own-mask read beat do-nothing? W0 and A as control, then N and Z on M and W. Peak GPU ~44 GB of the L40's 46.3, as job 51161 plus a transient of δW beside W0 + δW.
# Submit from the clone root: sbatch cluster/qwen_masked_read_amend2.sbatch
#SBATCH --job-name=commute-qwen_masked_read_amend2
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
.venv/bin/python experiments/qwen_masked_read/amend2.py --decomp "$COMMUTE_QWEN_DIR" "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
