#!/bin/bash
# E309 (cell P2.79): the reading refactor's old-code reproduction, gpu part (experiments/reading_refactor/old.py).
# Every reading the plan names, computed through the old experiments' own functions, written to
# experiments/reading_refactor/results/old_<TAG>.json. Two runs on the same card type (TAG A, then B) give the old code's
# spread. Nothing is trained; nothing large is written.
#   Submit from the root of a checkout of the HARNESS commit (old.py's commit on p2-79-refactor, before move 1 changed
#   src/), e.g. the worktree the writer used:
#   cd /mnt/nw/home/e.agnew/commute-agents/refactor-lead/.claude/worktrees/agent-a1f766142cd913b66
#   sbatch --comment='E309' experiments/reading_refactor/old.sbatch A
#   sbatch --comment='E309 stage: old-B floor' experiments/reading_refactor/old.sbatch B
# Inputs on NFS, read only (linked by fleet/link-data): artifacts/vpd (model), artifacts/sets (cached Pile rows), the
# saved updates and windows under experiments/*/results/large. The job prints "OLD <TAG> gpu DONE; errors: ..." at the
# end; a reading that raised is written as _error/<group> and the job exits 1 (the other readings still ran).
# Expected on an L40: about 40-50 min (README of the estimate: experiments/reading_refactor/NOTES.md).
#SBATCH --job-name=commute-e309-old
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:45:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
TAG=${1:?usage: sbatch old.sbatch TAG}
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the checkout's root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS tag=$TAG commit=$(git rev-parse HEAD)" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/reading_refactor/old.py --tag "$TAG" --part gpu
