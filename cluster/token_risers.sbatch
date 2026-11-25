#!/bin/bash
# E308 (cell P2.81, descriptive): token risers on the Fig 5 KL-hi states of lowercase, arXiv, Swahili, Yoruba
# (experiments/token_risers/run.py header). Rules: experiments/token_risers/results/predictions.json (committed first).
#   cd <clone> && ~/commute-agents/fleet/link-data .   (results/large must be the shared symlink)
#   sbatch --comment=E308 cluster/token_risers.sbatch
# Expected on an L40: ~1-2 min set-up, each task's rows built as base_loss.py builds them (arXiv and the wikis stream
# from HF), 13 forward reads of 64 x 512 tokens (base + 3 states per task, + one zero-update read) at seconds each.
#SBATCH --job-name=commute-token-risers
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; HF caches in /ephemeral/$USER (env.sh)
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/token_risers/run.py
