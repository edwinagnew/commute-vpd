#!/bin/bash
# E62 Amendment 3: a read of E62's six saved states, no training. Does head (2,4) act through the masks? Γ is read again with
# the head off. Plan: reports/plans/2026-11-02_plan_e62-copying-heads.md, "Amendment 3 (2026-11-02)".
# Rules: experiments/induction_cue/results/predictions_amend3.json (amend3.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/induction_cue_amend3.sbatch
# Exit 1 at a missed control: row 1 (the head off reaches Γ as filed, the frozen masks reproduce amend2.json, the zero update
# reads 0), row 2's re-read of the frozen path.
# Inputs (read): experiments/induction_cue/results/{run,amend2}.json, results/large/{windows,heads}.pt, {copy,lookup}_s{0,1,2}.pt
# (sha256 checked against run.json), experiments/transport_test/results/large/alive.pt, the VPD artifacts.
# Writes experiments/induction_cue/results/amend3.json (refuses if it exists) and the log, results/large/amend3.out. Deletes
# nothing, commits nothing. Expected under 10 min on a debug L40 (Amendment 2, with more forwards, took about 4.5 min).
#SBATCH --job-name=commute-e62-amend3
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
.venv/bin/python experiments/induction_cue/amend3.py "$@" 2>&1 | tee "$large/amend3.out" || rc=$?
echo "=== FINISHED E62 Amendment 3 (exit $rc) — results in $COMMUTE_REPO/experiments/induction_cue/results/amend3.json ==="
exit $rc
