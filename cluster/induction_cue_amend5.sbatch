#!/bin/bash
# E62 Amendment 5: W0 only, no training, no state read. At the second A, with W0's own mask, where does the masked forward's copy run, at the level of heads?
# Plan: reports/plans/2026-11-02_plan_e62-copying-heads.md, "Amendment 5 (2026-11-02)". Rules: experiments/induction_cue/results/predictions_amend5.json
# (amend5.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/induction_cue_amend5.sbatch
# Exit 1 at: stage a's control (head off with every mask at 1 falls under a quarter of 4.817, ablation_does_not_transfer), stage a's re-read of amend4.json,
# stage c's re-read of the labelled head. A precision move in stage b, or two precisions disagreeing, is a LOUD line and no stop.
# Inputs (read): experiments/induction_cue/results/{run,amend4}.json, results/large/{windows,heads}.pt (windows' sha256 checked against run.json),
# experiments/transport_test/results/large/alive.pt, the VPD artifacts.
# Writes experiments/induction_cue/results/amend5.json after each stage (refuses if it exists), results/large/amend5.out, results/large/amend5_vectors.pt
# (`a`, `a_half2`, S_old, before anything can stop) and results/large/amend5_windows.pt (every read per window, about 170 x 200 float64, under 1 MB).
# Deletes nothing, commits nothing.
# Time: expected about 2 min. Amendment 4's job took 34 s to its stage 1 (job 48945: 30 s to load, 4 s for two Γ passes and nine reads, about 13 reads' worth).
# This job is about 170 reads, so about 1 min of reads after the same load. The limit is 10 min, for a queue slot and to catch a hang.
#SBATCH --job-name=commute-e62-amend5
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
large=experiments/induction_cue/results/large
if [ -e "$large/amend5.out" ]; then echo "$large/amend5.out exists: move it aside, this run deletes nothing"; exit 1; fi
rc=0
.venv/bin/python experiments/induction_cue/amend5.py "$@" 2>&1 | tee "$large/amend5.out" || rc=$?
echo "=== FINISHED E62 Amendment 5 (exit $rc) — results in $COMMUTE_REPO/experiments/induction_cue/results/amend5.json ==="
exit $rc
