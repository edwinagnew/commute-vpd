#!/bin/bash
# E62, the copying-heads test: teach the 67M's in-context copying at a new marker beside a lookup at the same marker, label
# by switching off W0's copying heads (no VPD), then read r on both. Plan: reports/plans/2026-11-02_plan_e62-copying-heads.md.
# Rules: experiments/induction_cue/results/predictions.json (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/induction_cue.sbatch
# One job, exit 1 at the first missed control: (a) the heads and their ablation in W0, alone; (b) W0's prior, zero update,
# the per-window target path against E54's lookup path; (c) six fine-tunes; (d) the label; (e) r; the score.
# Inputs (read): experiments/keydependent_elicit/results/large/{windows_elicit,elicit_s0}.pt,
# experiments/transport_test/results/large/alive.pt, the VPD artifacts, the mined Pile rows.
# Writes experiments/induction_cue/results/run.json (small) and, in experiments/induction_cue/results/large/ (a link into
# ~/commute-data/results-large/induction_cue, made here if missing): windows.pt, heads.pt, {copy,lookup}_s{0,1,2}.pt (~113 MB
# each); the log tees to results/large/run.out. Commits nothing.
# Expected 8-15 min on a debug L40, estimated from E58 Amendment 7 (10 fine-tunes and reads in 13-17 min); host RAM peak ~5 GB.
#SBATCH --job-name=commute-e62-induction-cue
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
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/induction_cue/results/predictions.json || { echo "file the predictions first" >&2; exit 1; }
large=experiments/induction_cue/results/large
mkdir -p "$HOME/commute-data/results-large/induction_cue"   # also mends a dangling link's target
[ -L "$large" ] || [ -d "$large" ] || ln -s "$HOME/commute-data/results-large/induction_cue" "$large"
rc=0
.venv/bin/python experiments/induction_cue/run.py "$@" 2>&1 | tee "$large/run.out" || rc=$?
echo "=== FINISHED E62 (exit $rc) — results in $COMMUTE_REPO/experiments/induction_cue/results/ ==="
exit $rc
