#!/bin/bash
# E179 (P3.9): qwen_library_share/run.py on the frozen-neuron decomposition p-015e0710 on one L40; arguments pass to
# run.py (e.g. --run 1). Used for the 10-min probe of the controls after the MLP-only weight-swap fix:
#   cd ~/commute-agents/<role> && sbatch -t 00:10:00 --comment='E179 probe: controls after the MLP-only fix' \
#     cluster/qwen_library_share_frozen.sbatch --run 1
# GPU ~38 GiB, host RAM ~75 GB peak (E179 run 1 on a pod, 2026-11-13). States are fetched from W&B.
#SBATCH --job-name=commute-qwen_library_share_frozen
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:10:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
trap 'echo "=== FINISHED (exit $?) ==="' EXIT
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/python experiments/qwen_library_share/run.py "$@" --decomp "$COMMUTE_QWEN_FROZEN_DIR" \
  --fetch-to /ephemeral/$USER/e179_states
