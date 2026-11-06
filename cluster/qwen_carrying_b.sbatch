#!/bin/bash
# E102 (qwen_carrying) stage B's scoring job on one cluster L40, debug QoS, $0: experiments/qwen_carrying/maximise_b.py.
# Rules: experiments/qwen_carrying/results/predictions_stageB.json (early_checks). Submit from the clone root once every
# maximiser file is in results/stage_b/maximiser/:
#   sbatch cluster/qwen_carrying_b.sbatch
# Time: the model load (about 1 min, as stage A's jobs), E0 (16 rows of 512), C2 (about 8,000 windows of 41 tokens) and
# the agents' texts (about 8,000 of 10-60 words), batches of 16: a few minutes; the 15-min limit stops a hang.
# --mem: the fp32 model is 6.9 GB in host memory while it loads; tops.safetensors 45 MB.
# Exit 0 done; 1 a stop (E0 or C2 missed; results/stage_b/max_scores.json holds what was read) or a refusal.
#SBATCH --job-name=commute-qwen_carrying_b
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
set +e
.venv/bin/python experiments/qwen_carrying/maximise_b.py "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
