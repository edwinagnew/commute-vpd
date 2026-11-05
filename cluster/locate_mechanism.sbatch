#!/bin/bash
# E96 stages 0 and 1: name what a 67M fine-tune runs through, on E62's six saved states, no training. Plan:
# reports/plans/2026-11-05_plan_e96-locate-mechanism.md. Rules: experiments/locate_mechanism/results/predictions.json (run.py
# refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/locate_mechanism.sbatch
# Exit 1 at stage 0's miss. A failed control is not an exit code: read "outcome" in score.json.
# Inputs (read): experiments/induction_cue/results/run.json, predictions.json, results/large/windows.pt, {copy,lookup}_s{0,1,2}.pt
# (sha256 checked against run.json), the VPD artifacts, cached Pile rows. Never heads.pt.
# Writes experiments/locate_mechanism/results/{stage0,nominators,sweep,score}.json, means.pt, mlp_scores.pt, run.out; refuses if
# stage0.json exists. Deletes nothing, commits nothing. Expected about 5 min on a debug L40 (under 1,500 reads of 200 windows at
# E62 Amendment 2's ~0.1 s each, plus load; the plan says under 10).
#SBATCH --job-name=commute-e96-locate
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
.venv/bin/python experiments/locate_mechanism/run.py "$@" 2>&1 | tee -a experiments/locate_mechanism/results/run.out || rc=$?
echo "=== FINISHED E96 (exit $rc) — results in $COMMUTE_REPO/experiments/locate_mechanism/results/ ==="
exit $rc
