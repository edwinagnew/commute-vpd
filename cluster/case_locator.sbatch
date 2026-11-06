#!/bin/bash
# E114 case_locator: E97's locator (removal at head and MLP grain) and E100's carry / own / graft on E112's four saved
# case-label states (real and twin, seeds 0 and 1). No training. Plan: reports/plans/2026-11-06_plan_e114-locator-case-label.md.
# Rules: experiments/case_locator/results/predictions.json (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/case_locator.sbatch
# Stage 0 (alone, first): W0's held-out accuracy re-reads E112's 0.520625 / 0.496875; each state's gap gain on the fit
# windows re-reads E112's stage 2 gain within 0.02; all 24 reverted reads G 0; E97's two one-matrix plants (blocks 3, 1)
# read exactly at the edited part. Exit 1 on any miss. Stage 1: every state's reads, the located rule (real vs twin).
# Each stage prints "STAGE <n> DONE". A leftover stage0.json without status "pass" makes the run refuse: move it aside.
# Inputs (read only, sha256 checked): ~/commute-data/results-large/case_label/{task.pt,dW_{real,twin}_s{0,1}.pt}
# (E112's stage1.json), experiments/locate_mechanism/results/means.pt. Writes experiments/case_locator/results/stage{0,1}.json
# and run.out. Deletes nothing, commits nothing.
# Expected about 10 min on a debug L40: about 0.2 s per read of 1,024 x 41 tokens (E97: 0.07 s per forward of 399 x 41,
# three forwards per read); stage 0 about 3 min (two plants of ~250 reads each, the guards), stage 1 about 5.5 min (4
# states of ~290 reads, the held-out removal beside), start-up and checks about 1.5 min. --mem 16G as E112's job (host
# holds four 57 MB states in fp32 and task.pt).
#SBATCH --job-name=commute-e114-locator
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
.venv/bin/python experiments/case_locator/run.py 2>&1 | tee -a experiments/case_locator/results/run.out || rc=$?
echo "=== FINISHED E114 (exit $rc) — results in $COMMUTE_REPO/experiments/case_locator/results/ ==="
exit $rc
