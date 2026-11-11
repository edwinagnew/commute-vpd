#!/bin/bash
# Qwen instrument check, step 0 (cell P3.X, E###): three tables from saved files on the new decomposition p-9707d6ff.
# Plan: reports/plans/2026-11-11_plan_qwen-instrument-check-step0.md. CPU only, no GPU, no model forward.
#   cd ~/commute-agents/qwen-lead
#   sbatch --comment='E### step 0' cluster/qwen_instrument_check_step0.sbatch
# Reads: the components stacks of ~/commute-data/artifacts/qwen3-1.7b-p-9707d6ff-torch one layer slice at a time,
# 15 saved updates (~65 GB on NFS, one matrix at a time), the Qwen3-1.7B bf16 weights (3.4 GB) and stored JSONs.
# Time, measured 2026-11-11 on the worker with 8 CPUs: a 6144-side gram + eigh in fp64 takes 18 s, a 2048-side one 1 s;
# 84 MLP matrices need one of each, so ~30 min of eigh, plus ~10 min for the 112 attention matrices and the reads.
# A resubmit skips the matrices already in results/step0_sites.jsonl. Host memory: ~2 GB per matrix (fp64 alive
# vectors, one 6144x6144 gram and its eigenvectors, one Gaussian draw); 10G leaves room.
# The log prints TABLE 3 in its first seconds, one line per matrix, then TABLE 1, TABLE 2 and "STAGE 0 DONE".
#SBATCH --job-name=commute-qwen_instrument_check_step0
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=01:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=""
.venv/bin/python experiments/qwen_instrument_check/step0.py
