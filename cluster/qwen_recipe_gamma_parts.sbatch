#!/bin/bash
# E61 (qwen_recipe), side check: how much of Qwen's g is the readout, and how much a moved V changes g, on one cluster L40:
# experiments/qwen_recipe/gamma_parts.py. Rules: experiments/qwen_recipe/results/predictions_gamma_parts.json. Submit from the
# clone root, in this order; not submitted by the code writer:
#   sbatch cluster/qwen_recipe_gamma_parts.sbatch W0        # about 8 min, the control; every state refuses without it
#   sbatch cluster/qwen_recipe_gamma_parts.sbatch gold_s0   # about 20 min
#   sbatch cluster/qwen_recipe_gamma_parts.sbatch gauss     # about 22 min
# Optional, the same three with --gamma bf16 appended: Delta g on the Gamma path E57 and checks_k10 read, 6 min quicker each
# since the splits exist by then. e40 and a2long_s0 also run; no one has asked.
# Each state's first job makes its fp32 split, results/large/checks/parts_<state>_X.safetensors, 5.6 GB (the saved <state>_X
# files are bf16, and refused). To save 6 min, if Job S has already written size_<state>_b1.0_X.safetensors, hard-link it to
# that name before submitting. Remove parts_*_X.safetensors afterwards.
# Time: load 2 min, Sigma 171 s, split 355 s, Gaussian built 90 s (checks_k10_* logs). Rows are an estimate of 10 s per row per
# pass, the first job measures it (`[row 0 of W0]`). Exit 0 done; 1 a stop (results/gamma_parts_stopped.json, every state refuses
# until it is removed) or a refusal; 3 the 25-min budget was reached: resubmit the same command, the split is saved.
# Memory: GPU about 40.5 GiB (target 6.4, fp32 Gamma 31.9, one row's readouts 1.1; V on the host a site at a time), a stop at 44.
# Host about 70 GB: components 19.5, alive copies 14, two weight sets 11, X 5.6, dV 7, Sigma 4, the Gaussian's build 8.
#SBATCH --job-name=commute-qwen_recipe_gamma_parts
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
state=${1:?usage: sbatch cluster/qwen_recipe_gamma_parts.sbatch W0|gold_s0|gauss|e40|a2long_s0 [--gamma bf16]}
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
export PYTHONPATH=$COMMUTE_REPO/experiments/qwen_reading:$PYTHONPATH   # E57's lab.py and score.py
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_recipe/gamma_parts.py --state "$state" "${@:2}"   # the decomposition: COMMUTE_QWEN_DIR, env.sh
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
