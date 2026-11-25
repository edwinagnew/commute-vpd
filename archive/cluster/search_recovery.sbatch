#!/bin/bash
# E188 (cell P1.21): oracle check and repaired search on E184's built items. Plan:
# reports/plans/2026-11-16_plan_search-recovery.md. Filed rules: experiments/search_recovery/results/predictions.json (the
# run refuses unless it is committed unchanged). One L40 per job, debug QoS, $0. Every item saves as it goes and resumes; a
# job near 26.5 min exits 3, so submit through fleet/resubmit from the clone root:
#   fleet/resubmit --max 8 --comment='E188 search recovery' cluster/search_recovery.sbatch
# Exit 0 done, 1 a stop (a control or an exact form misses, a KL that does not repeat E184's), 3 more to do.
# Time on an L40 (from E184's run 1 timings, cluster/simplest_intervention.sbatch): stage 1 ~45 min (set-up ~1 min; per item
# ~0.5 min to prepare, up to 8 L-BFGS form fits of ~30 s, a 400-step rank-one refit ~1 min on the fine-tunes), so two slots;
# stage 2, only if some miss is put down to the search, ~10 min per control and ~20 min per item at worst.
# Memory: card ~10 GB as E184 (plus the scoring windows' standardised reads, ~0.1 GB); host ~6 GB, --mem 16G.
#SBATCH --job-name=commute-e188
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --comment=E188

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/search_recovery/run.py
