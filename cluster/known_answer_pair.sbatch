#!/bin/bash
# E103 (known_answer_pair) on one cluster L40, debug QoS, $0. Plan reports/plans/2026-11-06_plan_e103-known-answer-pair.md;
# rules experiments/known_answer_pair/results/predictions.json. Two jobs, in order, from the clone root:
#   sbatch cluster/known_answer_pair.sbatch run       # stages 0, T (C1 at the matched step), G
#   sbatch cluster/known_answer_pair.sbatch neurons   # after the first ends with exit 0: C2, then E102's readings
# Time on an L40. run: about 15 min. Stage 0 as E84's (83 s, experiments/qwen_tofu_forms/results/run.log:6); 335 steps at
# 0.9-1.4 s a step (E91's retrain, experiments/qwen_size_grid/results/run.log:61-64; E81's 1,250 steps in 1,793 s,
# experiments/qwen_tofu/results/run.log:25,87), so 5-8 min, plus about 30 candidate gain reads and two 5.64 GB saves;
# G 4 states at 24 s a load (qwen_size_grid/results/run.log:5-6) plus a 868-option read each. neurons: about 10 min,
# 7 loads and 10 reads of about 20 s on 128 rows (experiments/qwen_carrying/results/run.log:32). Limit 35 min each.
# --mem: E91's job, the same training and loads, peaked at 10.5 GB host (cluster/logs/commute-qwen_size_grid-52219.out:67);
# E102's second job at 10.5 GB (commute-qwen_carrying-53029.out:29). Exit 0 done; 1 a stop (results/stopped.json).
#SBATCH --job-name=commute-known_answer_pair
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:35:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
case "${1:-}" in
  run|neurons) ;;
  *) echo "give run or neurons"; exit 1 ;;
esac
set +e
.venv/bin/python "experiments/known_answer_pair/$1.py"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
