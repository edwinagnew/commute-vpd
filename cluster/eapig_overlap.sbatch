#!/bin/bash
# P2.69 eapig_overlap (E___): every stage in one debug job (stage 0 stops it on an adapter miss; later stages never stop).
# Needs the second venv first: sbatch --comment='E___ venv' cluster/eapig_overlap_venv.sbatch
# Usage (from the clone root): sbatch --comment='E___ eapig_overlap' cluster/eapig_overlap.sbatch [--stages 0 1 2 3 4]
# Scratch (native states, staged eap/, windows, native logits) is node-local /ephemeral, deleted when the job ends.
#SBATCH --job-name=commute-eapig-overlap
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/eapig_overlap/results/predictions.json || { echo "no predictions"; exit 1; }
test -x "$HOME/commute-data/venvs/fc310-fdaeeaa/bin/python" || { echo "no venv: sbatch cluster/eapig_overlap_venv.sbatch first"; exit 1; }
SCRATCH=/ephemeral/$USER/eapig_overlap/$SLURM_JOB_ID
LOG=experiments/eapig_overlap/results/run_$SLURM_JOB_ID.log
( .venv/bin/python experiments/eapig_overlap/run.py --scratch "$SCRATCH" "$@"; c=$?; echo "=== FINISHED (exit $c) — results in $COMMUTE_REPO/experiments/eapig_overlap/results ==="; exit $c ) 2>&1 | tee -a "$LOG"
code=${PIPESTATUS[0]}
rm -rf "$SCRATCH"
exit "$code"
