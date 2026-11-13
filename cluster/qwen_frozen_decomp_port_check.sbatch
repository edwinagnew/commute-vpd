#!/bin/bash
# E175 (P3.9): port check of the frozen-neuron decomposition p-015e0710 in commute.qwen_vpd, and the static count
# against p-9707d6ff (experiments/qwen_frozen_decomp/port_check.py). Run after convert and jax_reference.
#   cd ~/commute-agents/<role> && sbatch --comment='E175 stage: port check' cluster/qwen_frozen_decomp_port_check.sbatch
# Results: experiments/qwen_frozen_decomp/results/port_check.json, one key per stage as it ends ("STAGE x DONE").
# GPU memory (L40, 45 GiB): target fp32 7.6 + neuron components fp32 15.8 + Γ bf16 ~14.7 = ~38 GiB, then p-9707d6ff
# (components bf16 ~9, Γ bf16 ~16) after the frozen one is freed. Time: loads ~5-10 min from NFS, ~520 Γ rows, ~20 min.
#SBATCH --job-name=commute-qwen_frozen_decomp_port_check
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
trap 'echo "=== FINISHED (exit $?) ==="' EXIT
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/python experiments/qwen_frozen_decomp/port_check.py
