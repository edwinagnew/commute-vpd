#!/bin/bash
# qwen_port: the authors' JAX code (param-decomp 177d0cce8) on fixed tokens, as the reference the torch port must
# match. Plan: reports/plans/2026-10-27_plan_qwen-port-and-baseline.md, item 3.
#   cd ~/commute-agents/<role>
#   sbatch cluster/qwen_jax_reference.sbatch
# Needs: the JAX CUDA venv (cluster/qwen_jax_env.sh), experiments/qwen_port/results/tokens.json (make_tokens.py), the
# checkpoint at ~/commute-data/artifacts/qwen3-1.7b-p-e9fcffd5 and the HF snapshot of Qwen/Qwen3-1.7B in
# ~/.cache/huggingface/hub (the JAX loader reads only HF_HUB_CACHE, never the network).
# Results: results/jax_reference.json after every stage, big arrays in results/large/ (~4 GB, mostly fp32 logits).
# Expected on one L40: restore from NFS ~3-6 min (27 GB in bf16), compiles ~1 min each, target + taps + Γ + masked
# forwards on 8 rows ~2 min, adversary 3 seeds x 20 steps on 4 x 512 tokens ~2 min: ~10-15 min; the limit gives 2x.
# GPU memory: ~31 GB resident (target 3.4 + components 9.8 + Γ 17.1) plus ~4-8 GB per stage.
#SBATCH --job-name=commute-qwen_jax_reference
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub              # snapshot 70d244cc lives here, not in env.sh's HF_HOME
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95                      # one process; JAX's default 0.75 leaves ~11 GB idle
export JAX_COMPILATION_CACHE_DIR=/ephemeral/$USER/jax-cache
test -f experiments/qwen_port/results/tokens.json || { echo "write the token file first (make_tokens.py)"; exit 1; }
"$HOME/commute-data/param-decomp-177d0cce8/param_decomp/.venv/bin/python" experiments/qwen_port/jax_reference.py
