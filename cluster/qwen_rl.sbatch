#!/bin/bash
# E69 (qwen_rl) on one cluster L40: the rehearsal. Plan reports/plans/2026-11-03_plan_e69-grpo-loop.md; order and commands
# in experiments/qwen_rl/RUN_BRIEF.md. Submit from the clone root; needs WANDB_API_KEY in the submitting shell. The paid
# run is a pod, not this script.
#   sbatch cluster/qwen_rl.sbatch A0 tiny
#   sbatch cluster/qwen_rl.sbatch S full        # the full-size slice; the rate 0 keeps the cost rule off on the cluster
# Arguments after the size go to run.py (--upload-cap-gb 6, --lr-tag 3e-6).
# Exit 0 done; 1 a refusal; 2 an early check missed (results/<stage>.json "stopped"); 3 paused at a checkpoint (tiny
# only): resubmit the same command.
# --mem: E45's T peaked at 19 GB on the host with a resume (cluster/qwen_recipe_train.sbatch); δW in fp32 adds 5.6.
#SBATCH --job-name=commute-qwen_rl
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
stage=${1:?usage: sbatch cluster/qwen_rl.sbatch A0|S|R|N|F|score tiny|full}
size=${2:?usage: sbatch cluster/qwen_rl.sbatch A0|S|R|N|F|score tiny|full}
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_rl/run.py --stage "$stage" --size "$size" --usd-per-h 0 "${@:3}"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
