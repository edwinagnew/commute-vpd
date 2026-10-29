#!/bin/bash
# E44's input remade: E40's no-warm-up δW at steps 1, 10, 100 on one cluster L40, debug QoS, $0
# (experiments/qwen_maps/regen.py; brief experiments/qwen_maps/RUN_BRIEF_regen.md). Submit from the clone root:
#   sbatch --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True cluster/qwen_e40_regen.sbatch [--pause 50]
# Exit 0 done, and dW/SHA256SUMS written; 2 step 1 off E40 (STOP, results/regen.json "stopped"); 3 paused at a
# checkpoint (resubmit the same command). Files: $OUT/dW/step_{1,10,100}.safetensors, step_100_fp32.safetensors.
#SBATCH --job-name=commute-qwen_e40_regen
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
OUT=$HOME/commute-data/results-large/qwen_sft/nowarmup_regen
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_maps/regen.py --out "$OUT" "$@"
status=$?
if [ $status -eq 0 ]; then
  (cd "$OUT/dW" && sha256sum step_1.safetensors step_10.safetensors step_100.safetensors step_100_fp32.safetensors \
    > SHA256SUMS && cat SHA256SUMS && ls -l) || status=1
fi
echo "=== FINISHED (exit $status) ==="
exit $status
