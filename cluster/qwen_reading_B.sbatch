#!/bin/bash
# E57 (Q3) Stage B1 or B2 on one cluster L40, debug QoS ($0): experiments/qwen_reading/run.py --stage B1|B2. Plan:
# reports/plans/2026-10-30_plan_q3-reading-the-update.md, sections 4 (row 3), 6.3 and 7. Needs Stage A done. Submit
# from the clone root; not submitted by the code writer:
#   sbatch cluster/qwen_reading_B.sbatch B1     # think-on answers, ~20 min; then
#   sbatch cluster/qwen_reading_B.sbatch B2     # the T, L, C sets, ~25 min
# B1: bf16 generation, 2 batches of 64, max 2048 new tokens (KV ~16 GB); stops (exit 1) if over 25% of the traces do
# not close </think> after the first batch; resumes from results/large/B1/think.jsonl. B2: the longest row first; the
# GPU rule (44 GiB after it) stops with the length in the log. Host: the decomposition's load peak was 34.6 GB in
# Amendment 2, plus pinned W0 5.6 GB: 48G.
#SBATCH --job-name=commute-qwen_reading_B
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
stage=${1:?usage: sbatch cluster/qwen_reading_B.sbatch B1|B2}
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_reading/run.py --stage "$stage" --decomp "$COMMUTE_QWEN_DIR" "${@:2}"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
