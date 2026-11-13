#!/bin/bash
# E179 (P3.9): p-015e0710's references for the re-gating share reading: W0 g statistics, live-set count and the
# library floors K0 (experiments/qwen_library_share/run.py --run 0).
#   cd ~/commute-agents/<role> && sbatch --comment='E179 stage: refs' cluster/qwen_library_share_refs.sbatch
# Output: results/large/p-015e0710/gstats/W0_{fineweb,maths}.safetensors, results/p-015e0710/{refs,floors}.json and
# run.log. Expected on one L40, from E175's port check (job 59877: loads ~20 s, 256 Γ rows of 512 in ~1 min): g
# statistics on 256 rows of 512 ~2 min, floors on 5 x 128 rows of 256 (Γ + masked forward) ~3 min; ~10 min in all.
# GPU memory: ~37 GiB resident (target fp32, neuron components fp32, Γ bf16), as E175 measured.
#SBATCH --job-name=commute-qwen_library_share_refs
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
.venv/bin/python experiments/qwen_library_share/run.py --run 0 --decomp "$COMMUTE_QWEN_FROZEN_DIR"
