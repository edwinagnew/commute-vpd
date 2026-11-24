#!/bin/bash
# P2.73 weightwatch (E294): WeightWatch's detector and the score after Zhong on our items. Debug QoS, each job at most
# 30 min: run.py exits 3 past 25 min (wall_minutes) and a resubmit resumes by completed state, so chain it:
#   ~/commute-agents/fleet/resubmit --max 10 --comment='E294 weightwatch' cluster/weightwatch.sbatch
# (every stage in order: moments, stage0, trigger, directions, mfloor, items, summary). Sharded by state once the
# trigger stage has passed (directions) or the M floor is written (items), one chain per shard:
#   ~/commute-agents/fleet/resubmit --max 6 --comment='E294 weightwatch items 0/4' cluster/weightwatch.sbatch --stage items --shard 0 4
# Scratch (the rebuilt check's own-payload file) is node-local /ephemeral, deleted when the job ends.
#SBATCH --job-name=commute-weightwatch
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/weightwatch/results/predictions.json || { echo "no predictions"; exit 1; }
SCRATCH=/ephemeral/$USER/weightwatch/$SLURM_JOB_ID
LOG=experiments/weightwatch/results/run_$SLURM_JOB_ID.log
( .venv/bin/python experiments/weightwatch/run.py --scratch "$SCRATCH" "$@"; c=$?; echo "=== FINISHED (exit $c) — results in $COMMUTE_REPO/experiments/weightwatch/results ==="; exit $c ) 2>&1 | tee -a "$LOG"
code=${PIPESTATUS[0]}
rm -rf "$SCRATCH"
exit "$code"
