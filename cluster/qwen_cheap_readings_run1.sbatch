#!/bin/bash
# E162 (cell P3.6 run 1): the SVD column of the 19 saved Qwen3-1.7B updates beside a matched-norm random floor, and the
# top-set overlap's controls alone. Plan: reports/plans/2026-11-11_plan_qwen-cheap-readings.md; rules
# experiments/qwen_cheap_readings/results/predictions_run1.json (run1.py refuses unless it is committed unchanged).
#   cd <clone root> && ~/commute-agents/fleet/link-data . && sbatch --comment='E162 run1' cluster/qwen_cheap_readings_run1.sbatch
# Stage 0 (SVD known answers; the unchanged model against itself reads overlap exactly 1.0), stage 1 (two random draws at
# F_s0's and a2_s42's per-matrix norms, GPU), stage 2 (19 updates, one matrix at a time, CPU fp64 SVD); each prints
# "STAGE n DONE" after results/run1_stage<n>.json. A filed stop prints "STOP: ..." and exits 1. Exit 3: stage 2 projected
# past 100 min and stopped between states; resubmit the same command (it skips the states written, reruns stage 0).
# Inputs on NFS, read only: the 19 update files (15 named by experiments/qwen_instrument_check/step0.py update_files, E120's
# four at qwen_pass_pools/results/large/dW/{known,new}_s{0,1}/step_100.safetensors; 93 GB in all: 14 fp32 of 5.64 GB, 5 bf16 of 2.82 GB), Qwen3-1.7B from the HF
# cache, GSM8K test. Writes only experiments/qwen_cheap_readings/results/run1_stage{0,1,2}.json (under 1 MB).
# Time on an L40 with 16 CPUs, estimated: model and rows ~2 min; stages 0-1 five overlap reads of 103 rows (two fp32
# forwards each, E92 read a 5.64 GB state in 24 s and 557,000 tokens in 2.6-3.1 min with steering) and two 5.64 GB norm
# reads, ~3 min; stage 2 21 state-equivalents (19 updates and 2 Gaussian draws) of 196 fp64 SVDs, measured at 270 s each
# on 4 CPUs (scratch time_svd.out: 2.05 s per 6144x2048, 1.26 s per 2048x2048, 0.49 s per 1024x2048), so ~90-110 s on 16,
# plus ~24 s to read each fp32 file (7 min in all): ~40-50 min. Total ~50 min. GPU memory: the fp32 model 6.9 GB, one W of 6.9 GB, firing at
# answer positions ~3.5 GB per batch of 16: ~20 GB. Host: the model load ~7 GB, one fp64 matrix at a time after; 24G.
#SBATCH --job-name=commute-qwen_cheap_readings_run1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=01:50:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/qwen_cheap_readings/results/predictions_run1.json || { echo "file the predictions first"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
set +e
.venv/bin/python experiments/qwen_cheap_readings/run1.py
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
