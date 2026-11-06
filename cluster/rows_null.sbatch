#!/bin/bash
# E105 (rows_null) on one cluster L40, debug QoS, $0. Plan reports/plans/2026-11-06_plan_e105-rows-null.md; rules
# experiments/rows_null/results/predictions.json. One job, from the clone root:
#   sbatch cluster/rows_null.sbatch
# Stages C (C2n, then W0's sentence reader), then the four arms A42, A142, B42, B142, then the score. Time on an L40,
# from E103 Amendment 2's job 53316 (experiments/known_answer_pair/results/amend2.log): a candidate step with its
# 217-row read took 2.94 s (lines 18-49: steps 3 to 32 in 85 s), a step without a read 0.69 s (lines 52-64: steps 50 to
# 325 in 189 s); neuron reads 4-5 s on the real rows and 20 s on the TOFU rows (lines 77-80); a 5.64 GB save 21 s
# (line 50). Per arm about 32 candidate steps at about 1.8 s (half the rows read) + 32 steps at 0.69 s + 25 s of reads +
# about 10 s for the 2.8 GB bf16 save = about 2 min; the four arms about 8.5 min, stage C under 1 min, start-up about
# 1 min: about 11 min. Limit 30 min. --mem as the E103 jobs (peak host 10.5 GB, commute-known_answer_pair-53316.out).
# Exit 0 done; 1 a stop (results/stopped.json). A rerun after a lost job skips finished arms.
#SBATCH --job-name=commute-rows_null
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
set +e
.venv/bin/python experiments/rows_null/run.py
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
