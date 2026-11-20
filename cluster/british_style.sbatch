#!/bin/bash
# E233 (cell P2.48): British house style against a letter-scrambled twin. Plan
# reports/plans/2026-11-20_plan_p2-48-british-style.md; rules experiments/british_style/results/predictions.json (run.py
# refuses unless it is committed unchanged; `run` also unless it is filed). Submit from the clone root:
#   ~/commute-agents/fleet/link-data .
#   sbatch --comment='E233 rehearse: british-style' cluster/british_style.sbatch rehearse   (stage 0 in full, seed 0 cut to 40 steps)
#   sbatch --comment=E233 cluster/british_style.sbatch run                                  (scored: stage 0, seed 0 gain floor, seeds 1-2)
# Writes experiments/british_style/results/{data.json, <rehearse|scored>/stage0.json, stage1.json, states/*.json} as
# each ends; updates in bf16 (57 MB each) under results/large/<rehearse|scored>/states/. "RUN <mode> DONE" at the end;
# a filed stop prints "STOP: ..." and exits 1.
# Inputs: artifacts/sets/train_0_32768.npy and val_0_2000.npy (cached Pile rows), artifacts/spelling/<commit>/*.json
# (downloaded if missing), results/large/data.pt (built if missing), the 67M in artifacts/vpd/, E53's lowercase_s0
# state and E136/E98/update_svd records (stage 0).
#SBATCH --job-name=commute-british-style
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
export HF_HOME=$HOME/commute-data/cache/hf TMPDIR=$HOME/commute-data/cache/tmp; mkdir -p "$HF_HOME" "$TMPDIR"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/british_style/run.py "$1"
