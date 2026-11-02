#!/bin/bash
# E61 (qwen_recipe): the pipeline audit's five $0 checks on one cluster L40, debug QoS: experiments/qwen_recipe/checks.py.
# Rules: experiments/qwen_recipe/results/predictions_checks.json. Submit from the clone root, in this order; not
# submitted by the code writer:
#   sbatch cluster/qwen_recipe_checks.sbatch tiny                  # about 4 min
#   sbatch cluster/qwen_recipe_checks.sbatch k0                    # about 5 min
#   sbatch cluster/qwen_recipe_checks.sbatch sizes                 # about 15 min, no Γ
#   sbatch --mem=32G cluster/qwen_recipe_checks.sbatch gen         # 20-30 min, may take a second job; no Γ, no comps
#   sbatch cluster/qwen_recipe_checks.sbatch k10 --state gold_s0   # about 15-20 min each; also e40, a2long_s0, gauss
# Exit 0 done; 1 a stop (results/checks_stopped.json, every stage refuses until it is removed); 3 the 25-min budget was
# reached (resubmit the same command: the stage resumes from what is written). --mem: E57's stages with Γ peak at 72-93
# GB on the host (qwen_reading_C.sbatch); k10 holds 16 rows' masks on top, about 19 GB, so 128G. gen as E45's fp32 jobs.
#SBATCH --job-name=commute-qwen_recipe_checks
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
stage=${1:?usage: sbatch cluster/qwen_recipe_checks.sbatch tiny|k0|sizes|gen|k10 [--state NAME]}
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
export PYTHONPATH=$COMMUTE_REPO/experiments/qwen_reading:$PYTHONPATH   # E57's lab.py and score.py
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_recipe/checks.py --stage "$stage" --decomp "$COMMUTE_QWEN_DIR" "${@:2}"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
