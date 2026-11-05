#!/bin/bash
# E97 stage 3: the other saved pairs (rank-1, half-known and copy-24 against the 24-key cipher; ident32 against rand32) and
# the constant-answer tasks beside only (emoticon, two ladder rungs, against their same-windows null). No training.
# Plan: reports/plans/2026-11-05_plan_e97-locator.md. Rules: experiments/locator_table/results/predictions.json.
#   cd ~/commute-agents/<role> && sbatch cluster/e97_locator_stage3.sbatch     (after cluster/e97_locator.sbatch ended with exit 0)
# run.py refuses unless stage 0 passed and stage1.json holds a; stage 3 is skipped (logged) unless stage 1's control passed.
# Resumes from stage3.json: a resubmit reads only the states not yet written. Rewrites table.json from stages 1-3.
# Inputs (read, sha256 against each experiment's record): open_pair {elicit,cipher}_s{0,1,2}.pt, amend1/{rank1,copy}_s{0,1,2}.pt
# and their windows_*_trained.pt; regating_table {ident32,rand32}_s{0,1}.pt and windows; recruitment_test/states
# {emoticon,r1_t534_v369,r2_t359_v816}_{s0,s1,null}.pt; transport_test and task_ladder windows; the VPD artifacts.
# Writes experiments/locator_table/results/stage3.json, table.json, run.out. Deletes nothing, commits nothing.
# Expected about 8 min on a debug L40: 25 states of 73-106 windows of 41 tokens, ~190 reads each at about 0.05 s (E96's
# 0.07 s per read of 200 x 41, run.out lines 16-21), about 2 s per state file to check and load, start-up 1 min.
#SBATCH --job-name=commute-e97-locator-s3
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS stages=3" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
.venv/bin/python experiments/locator_table/run.py --stages 3 2>&1 | tee -a experiments/locator_table/results/run.out || rc=$?
echo "=== FINISHED E97 stage 3 (exit $rc) — results in $COMMUTE_REPO/experiments/locator_table/results/ ==="
exit $rc
