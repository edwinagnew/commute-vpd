#!/bin/bash
# E131: convert Goodfire's second Qwen3-1.7B decomposition (p-9707d6ff) from Orbax to fp32 safetensors, pads cut,
# V/U checked against the bundle's decomposition.safetensors. Plan: reports/plans/2026-11-07_plan_e131-new-decomposition.md.
#   cd ~/commute-agents/<role>
#   sbatch cluster/qwen_new_decomp_convert.sbatch
# Needs: the 7d2e5aee8 JAX venv (cluster/qwen_jax_env_7d2e5aee8.sh). CPU only, no GPU.
# Output: ~/commute-data/artifacts/qwen3-1.7b-p-9707d6ff-torch/ (~43 GB: 47 GB stored minus the pads; index.json,
# launch_config.yaml). The log ends "FINISHED <dir>"; exit 1 after writing if V/U differ from the bundle's.
# Expected: read 47 GB + write ~43 GB on NFS + two sha256 passes + the V/U comparison (~20 GB read): ~30-60 min
# (the p-e9fcffd5 conversion's cpu-run limit was 120 min). Host memory peak ~20 GB (blocks.w1 7.5 GB read whole,
# plus a <= 5 GB file group); 48G leaves room.
#SBATCH --job-name=commute-qwen_new_decomp_convert
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
trap 'echo "=== FINISHED (exit $?) — output in /mnt/nw/home/e.agnew/commute-data/artifacts/qwen3-1.7b-p-9707d6ff-torch ==="' EXIT
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
"$HOME/commute-data/param-decomp-7d2e5aee8-env/param_decomp/.venv/bin/python" -I experiments/qwen_new_decomp/convert.py
