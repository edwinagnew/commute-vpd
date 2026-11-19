#!/bin/bash
# E217 (cell P2.42, run 1): the definition read at a top-activating-neuron featurization of the 67M.
# Plan: reports/plans/2026-11-19_plan_topk-neurons.md. Rule: experiments/topk_neurons/results/predictions.json (run.py
# refuses unless it is committed unchanged). Submit from the clone root, stage 0 first, stage 1 after it passes:
#   sbatch --comment='E217 stage 0' cluster/topk_neurons.sbatch --stage 0
#   sbatch --comment='E217 stage 1' --dependency=afterok:<stage-0 id> --array=0-7%4 --time=01:45:00 \
#          --output=cluster/logs/%x-%A_%a.out cluster/topk_neurons.sbatch
# Stage 0 (alone; exit 1 on a miss): M's per-neuron percentiles on 400 Pile val rows, W0 at every band and the width,
# the floor (E208's rho.profile rerun on copy24_s0), the width control on all 20 items against E213. Prints "STAGE 0 DONE".
# Stage 1: array task T = band x seed (Config.tasks(): p97/3, width, p99/1, p90/10; seeds 0, 1), 10 items each, one
# file per item, skipped on a rerun, so a TIMEOUT resubmits cleanly (fleet/resubmit). Prints "STAGE 1 TASK T DONE".
# Inputs (NFS, read only): artifacts/vpd (the 67M), artifacts/sets (train_0_32768, train_32768_8192, val_0_600), E208's
# updates (experiments/{open_pair,regating_table,case_label,fresh_reuse_pair,built_control_sum,memorised_fact}/results/large),
# E208's parts npz (experiments/rho_saved_items/results/large/parts), E213's summary (committed).
# Expected on an L40: stage 0 about 10-15 min; each stage-1 task 35-75 min (E208's mechanism-replacement search took
# 1-3 min per order per item on an H100 on these windows; one order here, plus grades and attribution).
#SBATCH --job-name=commute-topk-neurons
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS array=${SLURM_ARRAY_TASK_ID:-} args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then set -- --stage 1 --task "$SLURM_ARRAY_TASK_ID"; fi
.venv/bin/python experiments/topk_neurons/run.py "$@"
