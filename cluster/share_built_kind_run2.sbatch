#!/bin/bash
# E202 (cell P2.33, second run): E200's share re-read on windows where the hand-made edit F raises its top token by at least
# 1 log-odds over M. Plan: reports/plans/2026-11-17_plan_share-built-kind.md (addendum "Second run"). Rules:
# experiments/share_built_kind/results/run2/predictions.json (the run refuses unless it is committed unchanged). One L40, debug QoS, $0.
# Submit from the clone root:
#   sbatch --comment='E202 share built kind big-gain windows' cluster/share_built_kind_run2.sbatch
# Stage 0 (identity) and stage 1 (positive control) stop with exit 1 on a miss; stage 2 stops if Always on, Cue or Table has
# under 32 big-gain windows. A rerun resumes stage 3 after the last state read.
# Inputs: as E200 (cluster/share_built_kind.sbatch) plus E200's experiments/share_built_kind/results/large/cue_s{0,1,2}.pt; no training.
# Writes: experiments/share_built_kind/results/run2/{stage0,stage1,stage2,stage3,rules}.json (small).
# Time on an L40: about 10 min (window scan up to the whole 100,000-window read half per pair ~3-6, controls ~1, 15 reads ~5).
#SBATCH --job-name=commute-e202
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/share_built_kind/run2.py
