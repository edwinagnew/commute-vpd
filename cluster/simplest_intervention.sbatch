#!/bin/bash
# E180 (cell P1.20): the fourth reuse wording read by a ranking of interventions on M's list (testing only, not adopted).
# Plan: reports/plans/2026-11-13_plan_simplest-intervention-ranking.md. Filed rules:
# experiments/simplest_intervention/results/predictions.json (the run refuses unless it is committed unchanged).
# One L40 per job, debug QoS, $0. The argument is the run: 1, then 2a 2b 2c 2d 3a 3b 3c 3d (in parallel). Every item saves
# after each greedy step and resumes; a job near 26.5 min exits 3, so submit through fleet/resubmit from the clone root:
#   fleet/resubmit --max 6 cluster/simplest_intervention.sbatch 1
# Runs 2* and 3* refuse (exit 1) until results/run1.json says "passed": true; then, each in its own chain,
#   fleet/resubmit --max 8 cluster/simplest_intervention.sbatch 2a      (2b 2c 2d 3a 3b 3c 3d likewise)
# Exit 0 done, 1 a stop (run 1's stop rule, a calibration or reload miss) or a refusal, 3 more to do.
# Time on an L40 (estimated, not measured: a forward of M ~34 us per 8-token window with logits at the last position only,
# from E173's ~6 s per 100,000 windows with all logits; a backward twice that): set-up ~1.5 min (E173's 55 s, the principal
# directions in run 1 ~10 s); an item's start ~0.5 min (rebuild, gap check, F's log-probabilities); a greedy step ~7-10 s
# (scoring pass with 2 Gauss-Newton samples ~1 s, pair scores ~1-2 s, 4 x 7 line-search points ~1 s, 50 Adam steps
# ~2.6 s, closure ~0.6 s); a rung-3 fit ~7 s (120 Adam steps). An item the search cannot close: per list 32 steps and 20
# fits, ~6-7 min, ~20 min for the three lists. An item closed at a low n: ~1-3 min. So run 1 ~25-30 min (L4 is the long
# one), runs 2a-2d and 3a-3d ~1-1.5 h each over 2-4 chained jobs.
# Memory: card ~12 GB (F's log-probabilities on 36,864 windows ~7.4 GB, the scoring matrices ~2.5 GB, one fine-tune copy);
# host ~6 GB as E173, --mem 16G.
# Reads E154's, E157's, E168's and E173's results JSONs and saved states, E137's pieces, the old tasks' delta-W and window
# files; writes only experiments/simplest_intervention/results/ (and results/large/pcs.pt in run 1).
#SBATCH --job-name=commute-e180
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --comment=E180

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/simplest_intervention/run.py --run "${1:?the run: 1, 2a-2d or 3a-3d}"
