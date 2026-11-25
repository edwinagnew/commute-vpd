#!/bin/bash
# E175 (P3.9): convert the frozen-neuron decomposition's CI function (p-015e0710) from Orbax to fp32 safetensors, and
# check the checkpoint's frozen V/U against the README table built from the HF weights
# (experiments/qwen_frozen_decomp/convert.py).
#   cd ~/commute-agents/<role> && sbatch --comment='E175 stage: convert' cluster/qwen_frozen_decomp_convert.sbatch
# Needs: the 7d2e5aee8 JAX venv. CPU only. Output: ~/commute-data/artifacts/qwen3-1.7b-p-015e0710-torch/ (Γ only,
# ~32 GB fp32; index.json, launch_config.yaml). Log ends "FINISHED <dir>"; exit 1 after writing if V/U differ.
# Expected, from E131's p-9707d6ff conversion (47 GB in ~30-60 min): Γ ~32 GB written plus the V/U read (~17 GB)
# and the HF read (~3 GB): ~30-45 min. Host memory peak ~20 GB (blocks.w1 7.5 GB read whole, plus a <= 5 GB file).
#SBATCH --job-name=commute-qwen_frozen_decomp_convert
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
trap 'echo "=== FINISHED (exit $?) ==="' EXIT
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
"$HOME/commute-data/param-decomp-7d2e5aee8-env/param_decomp/.venv/bin/python" -I experiments/qwen_frozen_decomp/convert.py
