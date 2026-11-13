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
# Time on an L40, from run 1 (jobs 59959, 59982): a forward of M ~0.1 ms per window, a backward twice that. Set-up ~1 min; an
# item's start ~0.5 min; a greedy step ~8 s (scoring 2,048 windows ~1 s, pair scores ~0.5 s, 4 x 4 line-search points on
# 1,024 windows ~1.7 s, 20 Adam steps ~3.2 s, closure on 16,384 windows ~1.7 s); rung 3 per list ~2.5 min (10 screening fits
# of 30 steps, 6 full fits of 80). An item no rung closes: per list at most 32 steps + rung 3, ~7 min, ~20 min for three
# lists; fewer where the greedy stops (3 steps each under 0.005). Run 1 ~25-35 min on an L40 (it resumes after exit 3);
# on an H100 pod ~2-2.5x faster: pass --wall-min 600 there so it never exits 3.
# Memory: card ~10 GB (F's log-probabilities on 34,816 windows ~7 GB, the scoring matrix ~0.5 GB, one fine-tune copy);
# host ~6 GB as E173, --mem 16G.
# Reads E154's, E157's, E168's and E173's results JSONs and saved states, E137's pieces, the old tasks' delta-W and window
# files; writes only experiments/simplest_intervention/results/ (pcs.pt there: the principal directions, fixed in run 1).
#SBATCH --job-name=commute-e180
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --comment=E182

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/simplest_intervention/run.py --run "${1:?the run: 1, 2a-2d or 3a-3d}"
