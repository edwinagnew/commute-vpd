#!/bin/bash
# E100 Amendment 2, job A: the sum of single reverts against a profile-matched written plant on each state's own site (rho),
# the bars and the kept rule; emoticon with E62's copying heads off; E99's class map and twin. No training.
# Plan: reports/plans/2026-11-05_plan_e100-amend2.md. Rules: experiments/fed_or_written/results/predictions_amend2.json
# (amend2.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/e100_amend2_a.sbatch
# Stages: 0 copy-24 s0's sum (0.39511) and the check's 4-MLP plant (0.6388) within 1e-3, exit 1 on a miss; 1 the 55 rows of
# table_amend1.json, bars, kept_rule; 2 the copying check and the class pair. Each prints "STAGE <n> DONE". A resubmit
# re-runs stage 0 and skips every row and state already in amend2.json (resume by state).
# Writes experiments/fed_or_written/results/amend2.json and amend2.out. Deletes nothing, commits nothing.
# Expected 10-15 min on a debug L40 (predictions_amend2.json cost.job_a: key-table states ~2-3 s each, E62 sites ~6 s, E53's
# whole rows ~70-110 s for each of 4 states, read last); peak host RSS in E100 was 6.1 GB, here one site's plant writes
# (~0.1 GB) and one dW at a time are added. If the 20 min run out, resubmit: it resumes.
#SBATCH --job-name=commute-e100-amend2-a
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
.venv/bin/python experiments/fed_or_written/amend2.py 2>&1 | tee -a experiments/fed_or_written/results/amend2.out || rc=$?
echo "=== FINISHED E100 Amendment 2 job A (exit $rc) — results in $COMMUTE_REPO/experiments/fed_or_written/results/ ==="
exit $rc
