#!/bin/bash
# E62 Amendment 4: a read of E62's six saved states, no training. Does the fine-tune's mask at the marker open what W0's own mask opens
# when W0 copies, and does the masked gain run through those subcomponents? Plan: reports/plans/2026-11-02_plan_e62-copying-heads.md,
# "Amendment 4 (2026-11-02)". Rules: experiments/induction_cue/results/predictions_amend4.json (amend4.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/induction_cue_amend4.sbatch
# Exit 1 at a missed control: stage 1 (W0's own mask at the second A, or k out of range), stage 3 (re-read of amend3.json, zero update).
# Stage 2's miss is a verdict, old_pieces_not_carrying, and no stop.
# Inputs (read): experiments/induction_cue/results/{run,amend3}.json, results/large/{windows,heads}.pt, {copy,lookup}_s{0,1,2}.pt
# (sha256 checked against run.json), experiments/transport_test/results/large/alive.pt, the VPD artifacts.
# Writes experiments/induction_cue/results/amend4.json (refuses if it exists), results/large/amend4.out and, at the end,
# results/large/amend4_vectors.pt (`a`, `a_half2`, S_old, each state's mask change; under 1 MB). Deletes nothing, commits nothing.
# Expected about 6 min on a debug L40 (Amendment 3, with fewer forwards, took 4.5 min).
#SBATCH --job-name=commute-e62-amend4
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
.venv/bin/python experiments/induction_cue/amend4.py "$@" 2>&1 | tee "$large/amend4.out" || rc=$?
echo "=== FINISHED E62 Amendment 4 (exit $rc) — results in $COMMUTE_REPO/experiments/induction_cue/results/amend4.json ==="
exit $rc
