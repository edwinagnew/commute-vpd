#!/bin/bash
# E228 (cell P2.44, widened 2026-11-20): the KL-form share and r on the nine built pairs' and four twins' 88 fine-tunes,
# three KL-matched random changes each and one structured control. Plan
# reports/plans/2026-11-20_plan_p2-44-random-kl-matched.md. Rules: experiments/random_kl_share/results/predictions.json.
# The scored run is planned on RunPod pods (Slurm down, user 2026-11-20); this script is the cluster form of the same job,
# for when the debug queue can take it (one shard per job, each under the 2 h debug cap at about 15-35 min a shard of 3):
#   sbatch --comment=E228 cluster/random_kl_share.sbatch states 0 3      (and states 1 3, states 2 3; any order)
#   then, no GPU: ~/commute-agents/fleet/cpu-run python3 experiments/random_kl_share/run.py score
# Writes experiments/random_kl_share/results/{zero_<i>_of_<n>.json,states/<state>.json}; a rerun skips finished states.
#SBATCH --job-name=commute-random-kl-share
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
export HF_HOME=$HOME/commute-data/cache/hf TMPDIR=$HOME/commute-data/cache/tmp; mkdir -p "$HF_HOME" "$TMPDIR"  # as E222: l40-worker's /ephemeral was full (2026-11-18)
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/random_kl_share/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/random_kl_share/run.py "$@"
