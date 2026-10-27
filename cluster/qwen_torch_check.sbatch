#!/bin/bash
# qwen_port torch check: the torch port of the Qwen3-1.7B decomposition (src/commute/qwen_vpd.py) against the JAX
# reference. Plan: reports/plans/2026-10-27_plan_qwen-port-and-baseline.md, Part 1 item 5 (pass rules: checks 0-4).
#   cd ~/commute-agents/<role>
#   sbatch cluster/qwen_torch_check.sbatch
# Inputs: experiments/qwen_port/results/tokens.json and the JAX reference (jax_reference.py's summary JSON and
# results/large/), the fp32 safetensors in ~/commute-data/artifacts/qwen3-1.7b-p-e9fcffd5-torch/ (COMMUTE_QWEN_DIR),
# HF Qwen/Qwen3-1.7B@70d244cc (downloaded to the node's /ephemeral HF cache if absent, ~4 GB).
# Output: experiments/qwen_port/results/torch_check.json after every stage; the log ends "FINISHED <path>".
# Expected on one L40: HF download + target ~2 min, check 0 ~1 min, U/V to host + Γ to GPU (54 GB read from NFS)
# ~5-10 min, checks 1-2 ~1 min, check 3 ~1 min, check 4 (3 seeds x 21 passes over 8 rows, fp32) ~4-6 min: ~15-20 min.
# Peak GPU memory ~34 GB (stages 1-2); host ~45 GB.
#SBATCH --job-name=commute-qwen_torch_check
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
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/qwen_port/results/tokens.json || { echo "missing results/tokens.json (make_tokens.py)"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/python experiments/qwen_port/torch_check.py "$@"
