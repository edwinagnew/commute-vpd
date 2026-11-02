#!/bin/bash
# E61 (qwen_recipe) Job R: fp32 fine-tunes on one cluster L40: experiments/qwen_recipe/train.py. Rules:
# experiments/qwen_recipe/results/predictions_train.json. Submit from the clone root, in this order; not submitted by the
# code writer. Exit 0 done; 1 an early stop (results/stopped.json, every stage refuses until it is removed) or a
# refusal; 3 paused, work saved: resubmit the same command.
# Debug QoS (arm 0, Stage W, each Stage G; W and G stop themselves at 25 min, exit 3):
#   sbatch --qos=debug --time=00:30:00 cluster/qwen_recipe_train.sbatch T --run arm0            # 8 min, exit 3 at step 5
#   sbatch --qos=debug --time=00:30:00 cluster/qwen_recipe_train.sbatch T --run arm0            # 8 min, steps 6-10
#   sbatch --qos=debug --time=00:30:00 cluster/qwen_recipe_train.sbatch W                       # 35-40 min, 2 jobs
# Normal QoS, this script's default; when it is not free, a 100-step arm fits debug in two jobs with --t-chunk 50:
#   sbatch cluster/qwen_recipe_train.sbatch T --run a1_3e-6                                     # 31 min
#   sbatch cluster/qwen_recipe_train.sbatch T --run a1_1e-6                                     # 31 min
#   sbatch cluster/qwen_recipe_train.sbatch T --run gold_1e-5                                   # 37 min
#   sbatch cluster/qwen_recipe_train.sbatch T --run a2_6e-6                                     # 46 min, exit 3 at step 150
#   sbatch cluster/qwen_recipe_train.sbatch T --run a2_6e-6                                     # 46 min, steps 151-300
#   sbatch --qos=debug --time=00:30:00 cluster/qwen_recipe_train.sbatch G --run a1_3e-6         # 10 min; a1_1e-6 the same
#   sbatch --qos=debug --time=00:30:00 cluster/qwen_recipe_train.sbatch G --run gold_1e-5       # 6 min
#   sbatch --qos=debug --time=00:30:00 cluster/qwen_recipe_train.sbatch G --run a2_6e-6         # 15 min
# Job S's small state, one saved state's reads and no verdict: G --run gold_1e-5 --step T (order: qwen_recipe_size.sbatch).
# --mem: E45's T peaked at 19 GB on the host with a resume (T_a2long_s0.peaks), fp32 generation at 11; 64G.
#SBATCH --job-name=commute-qwen_recipe_train
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
stage=${1:?usage: sbatch cluster/qwen_recipe_train.sbatch T|W|G [--run NAME] [--t-chunk N] [--step T]}
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_recipe/train.py --stage "$stage" "${@:2}"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
