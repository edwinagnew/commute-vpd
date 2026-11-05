#!/bin/bash
# E97 Amendment 1, job 1: stage 0 (exact edge checks, the fp32 floor measured first; stops on a miss), stage 1 (E62 copy
# and lookup: every edge exact, W0's circuits, cross-faithfulness, the band and the tree's outcome written before stage 2),
# stage 2 (lowercase and cipher on whole rows, IG with the top 50 exact; only on the tree's "copy high, lookup low").
# Plan: reports/plans/2026-11-05_plan_e97-amend1.md. Rules: experiments/locator_table/results/predictions_amend1.json
# (amend1.py refuses unless it and E97's predictions.json are committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/e97_amend1.sbatch
# Stage 3 is the second job: cluster/e97_amend1_stage3.sbatch, after this one ends with exit 0 and amend1_stage1.json "go" true.
# Exit 1 at a stage-0 miss (incl. the edge-sum check on E62, run in stage 1). An outcome is not an exit code: read
# amend1_stage1.json "outcome". The log prints "STAGE <n> DONE" only after amend1_stage<n>.json is complete, then "STAGE table DONE".
# A stage resumes from its file: if the time limit cuts stage 2, resubmit with `--export=ALL,STAGES=2`; finished states are kept.
# Inputs (read, sha256 against each experiment's record): as cluster/e97_locator.sbatch, plus E97's stage1.json, stage2.json.
# Writes experiments/locator_table/results/amend1_stage{0,1,2}.json, table_amend1.json, amend1.out. Deletes nothing, commits nothing.
# Expected about 25 min on a debug L40 (estimates, not measured; sources: E97's run.out, stage 0 25 s, stage 1 99 s for 6 x ~190
# reads of 200 x 41 tokens, stage 2 1282 s, i.e. ~1.2 s per read of 256 whole rows; an edge forward taken as ~2x a module read):
# start-up and sites 2 min; stage 0 about 2 min (two plants, 2 x 929 stacked exact patches on 75 windows, floors in fp64);
# stage 1 about 4 min (6 states x 955 exact patches on 200 windows cut to 21 tokens, stacked 3 per forward; 8 circuits;
# IG beside); stage 2 about 17 min (4 states x [IG 8 steps ~1 min, 52 exact ~1.2 min, own circuit ~1.5 min, 2 cross
# ~15 s] + W0's 2 circuits ~3 min). If the 30-min limit cuts stage 2, resubmit as above (about 5-10 min more).
#SBATCH --job-name=commute-e97-amend1
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
.venv/bin/python experiments/locator_table/amend1.py --stages "${STAGES:-0,1,2}" 2>&1 | tee -a experiments/locator_table/results/amend1.out || rc=$?
echo "=== FINISHED E97 Amendment 1 stages ${STAGES:-0,1,2} (exit $rc) — results in $COMMUTE_REPO/experiments/locator_table/results/ ==="
exit $rc
