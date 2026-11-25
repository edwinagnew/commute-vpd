#!/bin/bash
# E175 (P3.9): the authors' public library (param-decomp 7d2e5aee8) on the frozen-neuron decomposition p-015e0710 on
# the 8 qwen_port rows, the reference for port_check.py's stage 2 (experiments/qwen_frozen_decomp/jax_reference.py).
#   cd ~/commute-agents/<role> && sbatch --comment='E175 stage: jax reference' cluster/qwen_frozen_decomp_jax_reference.sbatch
# Needs: the 7d2e5aee8 JAX CUDA venv, the extracted bundle (read only), the HF snapshot in ~/.cache/huggingface/hub.
# Results: experiments/qwen_frozen_decomp/results/jax_reference.json after every stage; results/large/jax_gamma_row*
# (~2.6 GB: g of 84 sites for 8 rows in bf16).
# Expected on one L40, from E131's p-9707d6ff reference (~10-15 min): restore ~3-6 min, stages ~5 min.
# GPU memory: as p-9707d6ff's (~29 GiB load peak on a 45 GiB L40); the frozen stacks are the MLP ones at the same C.
#SBATCH --job-name=commute-qwen_frozen_decomp_jax_reference
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
trap 'echo "=== FINISHED (exit $?) ==="' EXIT
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_COMPILATION_CACHE_DIR=/ephemeral/$USER/jax-cache
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
"$HOME/commute-data/param-decomp-7d2e5aee8-env/param_decomp/.venv/bin/python" -I experiments/qwen_frozen_decomp/jax_reference.py
