#!/bin/bash
# E62 Amendment 2: a read of E62's six saved states, no training. Does the mask's gain on the copy map go through the same old
# head, (2,4), as the target model's? Plan: reports/plans/2026-11-02_plan_e62-copying-heads.md, "Amendment 2 (2026-11-02)".
# Rules: experiments/induction_cue/results/predictions_amend2.json (amend2.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/induction_cue_amend2.sbatch
# Exit 1 at a missed control: row 1 (top-1 through the masked forward), row 2 (the target share re-reads), row 3's re-read.
# Inputs (read): experiments/induction_cue/results/run.json, results/large/{windows,heads}.pt, {copy,lookup}_s{0,1,2}.pt
# (sha256 checked against run.json), experiments/transport_test/results/large/alive.pt, the VPD artifacts.
# Writes experiments/induction_cue/results/amend2.json (refuses if it exists) and the log, results/large/amend2.out. Deletes
# nothing, commits nothing. Expected under 10 min on a debug L40 (E62's whole job, with six fine-tunes, took 3 min 25 s).
#SBATCH --job-name=commute-e62-amend2
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
large=experiments/induction_cue/results/large
rc=0
.venv/bin/python experiments/induction_cue/amend2.py "$@" 2>&1 | tee "$large/amend2.out" || rc=$?
echo "=== FINISHED E62 Amendment 2 (exit $rc) — results in $COMMUTE_REPO/experiments/induction_cue/results/amend2.json ==="
exit $rc
