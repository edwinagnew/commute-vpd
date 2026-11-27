#!/bin/bash
# E317 (cell P2.85 run 2): the positive control for E314's error share. The reconstruction share (each of the 8
# block-output sites set to hat(o_FT) + e(o_cur), all at once) beside the error share read again on the same forwards,
# on E314's 48 states and their random updates. experiments/sae_proxies/complement.py (reuses run.py's Reader); rule
# experiments/sae_proxies/results/predictions_complement.json (refused unless committed unchanged; E314's
# predictions.json is required unchanged too).
#   cd ~/commute-agents/sae-lead && sbatch --comment=E317 cluster/sae_proxies_complement.sbatch
# Stage 0 (zero update, run-to-run floor, full patch) prints "STAGE 0 DONE"; stage 1 writes each state to
# results/results_complement.json and prints "STAGE 1 DONE"; a stop prints "STOP: ..." and exits 1; a rerun resumes.
# Inputs as cluster/sae_proxies.sbatch (model, E234's SAEs, the 48 update files, the windows).
# Expected on a debug card, from E314's job 69300 (7.3 min, peak 22.4 GB GPU, 6.1 GB host, including EV/FVU, the Pile
# reads and the 48 gains this job skips): this job runs 4 row forwards per update (E314: 3 plus a Pile one), so
# about 6 min; limit 60 min.
#SBATCH --job-name=commute-sae-proxies-complement
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --comment=E317
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/sae_proxies/results/predictions_complement.json || { echo "file the predictions first"; exit 1; }
rc=0
( .venv/bin/python experiments/sae_proxies/complement.py ) 2>&1 | tee -a experiments/sae_proxies/results/run_complement.log || rc=${PIPESTATUS[0]}
echo "=== FINISHED E317 (exit $rc) — results in $COMMUTE_REPO/experiments/sae_proxies/results/ ==="
exit $rc
