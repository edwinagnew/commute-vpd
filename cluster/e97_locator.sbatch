#!/bin/bash
# E97 stages 0-2: the plants (stage 0, exact, stops on a miss), copy against lookup (stage 1, the control; writes a to
# stage1.json), lowercase against cipher on whole rows (stage 2). No training. Plan: reports/plans/2026-11-05_plan_e97-locator.md.
# Rules: experiments/locator_table/results/predictions.json (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/e97_locator.sbatch
# Stage 3 is the second job: cluster/e97_locator_stage3.sbatch, after this one ends with exit 0.
# Exit 1 at a stage-0 miss, an all-reverted G off 0, or a degenerate a. An outcome is not an exit code: read stage1.json "outcome".
# A stage resumes from its file: if the time limit cuts stage 2, resubmit with `--export=ALL,STAGES=2`; finished states are kept.
# Inputs (read, sha256 against each experiment's record): induction_cue copy/lookup_s{0,1,2}.pt and windows.pt; performed_anchor
# dW_{lowercase,cipher}_s{0,1}_step300.pt, a0.json; open_pair/amend1 windows_copy_trained.pt and copy_s0.pt; locate_mechanism
# means.pt and stage2_stage0.json; cached Pile val rows 0-255; the VPD artifacts.
# Writes experiments/locator_table/results/stage{0,1,2}.json, table.json, run.out. Deletes nothing, commits nothing.
# Expected about 23 min on a debug L40, from E96's timings (run.out lines 16-21: 0.07 s per read of 200 x 41 tokens;
# stage2.out lines 12-16: about 0.6 s per read of 500 x 127, so about 1.2 s per read of 256 whole rows of 512): stage 0
# about 1.5 min, stage 1 about 2 min (6 states x ~190 reads), stage 2 about 18 min (4 states x ~190 reads at 1.2 s, W0's
# means reads, 1.5 min of 127-token windows beside), start-up 1 min.
#SBATCH --job-name=commute-e97-locator
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS stages=${STAGES:-0,1,2}" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
.venv/bin/python experiments/locator_table/run.py --stages "${STAGES:-0,1,2}" 2>&1 | tee -a experiments/locator_table/results/run.out || rc=$?
echo "=== FINISHED E97 stages ${STAGES:-0,1,2} (exit $rc) — results in $COMMUTE_REPO/experiments/locator_table/results/ ==="
exit $rc
