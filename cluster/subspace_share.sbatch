#!/bin/bash
# P2.87 (experiments/subspace_share): updates trained only inside fixed random subspaces on Fig `known`'s 8 reuse tasks,
# each by its task's fine-tune recipe, read by the log-odds re-gating share exactly as the fine-tunes are. Card
# experiments/subspace_share/card.md; settings and expected values experiments/subspace_share/results/predictions.json
# (run.py refuses unless it is committed unchanged and its "experiment" is claimed). The caller passes the E-number:
#   sbatch --comment=E### cluster/subspace_share.sbatch control       # alone, first; prints "STAGE 0 DONE"
#   sbatch --comment=E### cluster/subspace_share.sbatch lrgrid        # after control; prints "LRGRID DONE"
#   sbatch --comment=E### cluster/subspace_share.sbatch train e62/copy e46/ident32 e54/elicit c3/rank1
#   sbatch --comment=E### cluster/subspace_share.sbatch train c3/copy e58/rank1 e58/copy e53/lowercase
# The two train jobs may run in parallel. A stop prints "STOP: ..." and exits 1; a rerun skips finished items
# (results/updates/<task>_s<seed>.json).
# Inputs (NFS, read only): the 24 reuse fine-tunes named in known_3seeds/results/known3.json, their tasks' training and
# read windows, E53's cached Pile rows, the model in artifacts/vpd/. Writes results/control.json, results/lrgrid.json,
# results/updates/*.json (tens of KB each) and results/large/<task>_s<seed>.pt (Z, seeds, two bf16 deltas, ~60 MB each).
# Expected on a debug L40 (from measured runs): model load, SVDs and windows ~3 min (P2.82); control ~5 min (24 shares,
# 32 KL reads, 8 zero reads at ~2 s); a training of 300 steps ~20-30 s (known_3seeds job 66292: arm A 19-30 s, arm B
# 24-28 s) plus a held-out gain every step (~0.05-1 s each, so ~15 s to ~5 min with e53's longer windows) plus 2 reader
# reads (~10 s): lrgrid 3 trainings ~5 min; each train job 12 items ~15-25 min. Limit 30 min; a cut job resumes.
# Memory: E136's 20.4 GB host peak came from E53's gates (job 56944); 40G as random_logodds_share.sbatch.
#SBATCH --job-name=commute-subspace-share
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/subspace_share/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/subspace_share/run.py "$@"
