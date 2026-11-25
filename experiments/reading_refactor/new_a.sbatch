#!/bin/bash
# E309 (cell P2.79): the reading refactor's NEW-code reproduction, writer 2a's GPU readings (moves 2-4:
# experiments/reading_refactor/new_a.py --part gpu): the re-gating share in every variant, KL both directions, the
# geometric prior and the panel's share, gain, KL, prior, size and erank, through commute.share / kl / prior only.
# Writes experiments/reading_refactor/results/new_a.json (compare.py merges it with new_b.json). Nothing is trained;
# nothing large is written. Run on the same card type as old.sbatch (L40), after the old A/B batches.
#   Submit from the root of a checkout of the branch p2-79-refactor at or after this file's commit:
#   sbatch --comment='E309' experiments/reading_refactor/new_a.sbatch
# Inputs on NFS, read only (linked by fleet/link-data): artifacts/vpd (model), artifacts/sets (cached Pile rows), the
# saved updates and windows under experiments/*/results/large. The job prints "NEW_A gpu DONE; errors: ..." at the
# end; a reading that raised is written as _error/<group> and the job exits 1 (the other readings still ran).
# Expected on an L40: the same groups as old.sbatch's share_e136, share_e211, share_e140, prior_forward,
# kl_procedural and panel (without its steering and ADL): under 30 min; use old run A's `_seconds` for these groups.
#SBATCH --job-name=commute-e309-new-a
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the checkout's root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS commit=$(git rev-parse HEAD)" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/reading_refactor/new_a.py --part gpu
