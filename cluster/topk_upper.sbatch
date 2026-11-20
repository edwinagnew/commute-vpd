#!/bin/bash
# Cell P2.45 (was P2.42 run 2), E229: E217's top-activating-neuron featurization of the 67M re-read with the upper tail only.
# Plan: reports/plans/2026-11-20_plan_p2-42-upper-tail.md. Rule: experiments/topk_upper/results/predictions.json (run.py
# refuses unless it is committed unchanged). Submit from the clone root, stage 0 first, stage 1 after it passes:
#   sbatch --comment='E### stage 0' cluster/topk_upper.sbatch --stage 0
#   sbatch --comment='E### stage 1' --dependency=afterok:<stage-0 id> --array=0-5%4 --time=01:45:00 \
#          --output=cluster/logs/%x-%A_%a.out cluster/topk_upper.sbatch
# Stage 0 (alone; exit 1 on a miss): E217's saved percentiles checked by sha, the upper-only feature share on 400 Pile
# val rows, W0 at p97, p99, p90 and the width, the width control on all 20 items against E213 (bar from E217's floor).
# Prints "STAGE 0 DONE". Stage 1: array task T = band x seed (Config.tasks(): p97, p99, p90; seeds 0, 1), 10 items each,
# one file per item, skipped on a rerun, so a TIMEOUT resubmits cleanly (fleet/resubmit). Prints "STAGE 1 TASK T DONE".
# Inputs (NFS, read only): as E217 (cluster/topk_neurons.sbatch), plus E217's results/large/percentiles.pt, stage0.json
# and results/items (the score reads them beside run 2's).
# Expected on an L40: stage 0 about 3-5 min (E217's took 2.3 min, with the percentiles and the floor it no longer
# recomputes); each stage-1 task 50-95 min (E217's per-item times, 300-610 s at its bands).
#SBATCH --job-name=commute-topk-upper
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
.venv/bin/python experiments/topk_upper/run.py "$@"
