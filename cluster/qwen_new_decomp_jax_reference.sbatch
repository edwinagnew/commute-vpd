#!/bin/bash
# E131: the authors' public library (param-decomp 7d2e5aee8) on Goodfire's second Qwen3-1.7B decomposition
# (p-9707d6ff) on the 8 qwen_port rows, as the reference for control (b). Plan:
# reports/plans/2026-11-07_plan_e131-new-decomposition.md.
#   cd ~/commute-agents/<role>
#   sbatch cluster/qwen_new_decomp_jax_reference.sbatch
# Needs: the 7d2e5aee8 JAX CUDA venv (cluster/qwen_jax_env_7d2e5aee8.sh), the bundle at
# ~/commute-data/artifacts/p-9707d6ff-share/data/p-9707d6ff-share/ (read only), and the HF snapshot of Qwen/Qwen3-1.7B
# in ~/.cache/huggingface/hub (the JAX loader reads only HF_HUB_CACHE).
# Results: experiments/qwen_new_decomp/results/jax_reference.json after every stage; results/large/jax_*.safetensors
# (~3.7 GB: g for 8 rows in bf16 3.4 GB, taps 0.3 GB).
# Expected on one L40 (the p-e9fcffd5 reference, job 38812: load 43 s after a 3-6 min restore, target 18 s, taps 3 s,
# Γ 180 s incl. compiles, masked 29 s): restore ~3-6 min, stages ~5 min, ~10-15 min in all.
# GPU memory (GiB; L40 45.0): load peak ~29 as for p-e9fcffd5 (target 3.2 + restored 25, the prepare jits donate);
# resident ~28 (+ the sink logits, 2 KB); a batch of two rows of 512 adds ~2-4.
#SBATCH --job-name=commute-qwen_new_decomp_jax_reference
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
trap 'echo "=== FINISHED (exit $?) — results in $COMMUTE_REPO/experiments/qwen_new_decomp/results/jax_reference.json ==="' EXIT
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub              # snapshot 70d244cc lives here, not in env.sh's HF_HOME
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95                      # one process; JAX's default 0.75 leaves ~11 GB idle
export JAX_COMPILATION_CACHE_DIR=/ephemeral/$USER/jax-cache
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
"$HOME/commute-data/param-decomp-7d2e5aee8-env/param_decomp/.venv/bin/python" -I experiments/qwen_new_decomp/jax_reference.py
