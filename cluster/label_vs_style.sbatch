#!/bin/bash
# E107 (label_vs_style) on one cluster L40, debug QoS, $0. Plan reports/plans/2026-11-06_plan_e107-label-vs-style.md;
# rules experiments/label_vs_style/results/predictions.json. One job, from the clone root:
#   sbatch cluster/label_vs_style.sbatch
# Stages C (C2n, W0's true-sentence level, the label check on the wrong sentences), then WRONG42, WRONG142, then AUTH42,
# AUTH142, WORLD42, WORLD142, then the score. Time on an L40: E105's job 53480 (experiments/rows_null/results/run.log)
# took 30 s for C2n (lines 2-3) and 2.5 s for a 217-row sentence read (line 4); each 108/109-row arm 130 s (line 5 at
# 37.7 s to line 54 at 167.5 s), whole job 556 s (line 207). A 217-row arm, from E103 Amendment 2's job 53316
# (experiments/known_answer_pair/results/amend2.log): a candidate step with its 217-row read 2.94 s (lines 18-49), a step
# without 0.69 s (lines 52-64), a 5.64 GB fp32 save 21 s (line 50); neuron reads 25 s (E105 run.log lines 51-53). So a
# wrong arm is about 32 x 2.94 + 32 x 0.69 + 5 s of reads + 25 + 21 = 2.8 min; the two 5.5 min; the four topic arms about
# 130 s each, 8.7 min; stage C 36 s; start-up about 1 min: about 16 min. Limit 30 min. --mem as E105's (host peak
# 11.0 GB, E103's fp32 saves 10.5 GB). Exit 0 done; 1 a stop (results/stopped.json). A rerun after a lost job skips
# finished stages. To split (only if the queue makes one job risky): the wrong arms are done after about 7 min; a second
# submission of the same script runs the topic arms.
#SBATCH --job-name=commute-label_vs_style
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
.venv/bin/python experiments/label_vs_style/run.py
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
