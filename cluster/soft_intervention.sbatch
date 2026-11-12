#!/bin/bash
# E173 (cell P1.19): the third reuse wording read by interchange interventions (testing only, not adopted).
# Plan: reports/plans/2026-11-12_plan_soft-intervention-wording.md. Filed rules:
# experiments/soft_intervention/results/predictions.json (the run refuses unless it is committed unchanged).
# One L40 per job, debug QoS, $0. The argument is the run: 1a, 1b, 2, 3a or 3b. Submit from the clone root, run 1 first
# (jobs A and B, in parallel):
#   sbatch --comment=E173 cluster/soft_intervention.sbatch 1a
#   sbatch --comment=E173 cluster/soft_intervention.sbatch 1b
# Runs 2, 3a, 3b refuse (exit 1) until results/run1a.json and run1b.json both say "passed": true; then
#   sbatch --comment=E173 cluster/soft_intervention.sbatch 2      (and 3a, 3b, in parallel)
# Exit 1 = a stop (run 1's stop, an S0-S3 or reload check) or a refusal; the item's JSON or run<run>.json says which.
# Each item writes results/<item>.json when it finishes and is skipped on a resubmit.
# Time on an L40 (estimated from E168's ~6 s per pass over 100,000 windows; nothing is trained): set-up ~1 min; an item
# ~1-2 min (about 12 passes); 1a ~8 min (two calibrations), 1b ~7, 2 ~8, 3a ~6, 3b ~10 (the case pair's 800-word fits).
# Memory: card well under the L40's 44 GB (one fine-tune copy, batches of 2,048 windows of 8 tokens); host ~5 GB at the
# case pair's fits, --mem 16G as E168.
# Reads E154's, E157's and E168's results JSONs and saved fine-tunes, E137's pieces, the old tasks' delta-W and window
# files; writes only experiments/soft_intervention/results/.
#SBATCH --job-name=commute-e173
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --comment=E173

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/soft_intervention/run.py --run "${1:?the run: 1a, 1b, 2, 3a or 3b}"
