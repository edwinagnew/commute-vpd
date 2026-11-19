#!/bin/bash
# E210 (cell P2.37) stage b, one L40: 1M FineWeb + 1M LMSYS tokens through gemma-2-2b and gemma-2-2b-it (layer 13
# output), the crosscoder's FVE first (stop under 0.7), then our latent-scaling betas on the top 3,176 latents against
# feature_df. Plan: reports/plans/2026-11-18_plan_p2-37-crosscoder-replicate.md. Needs stage a passed and HF_TOKEN
# (Gemma-2 licence) exported in the submitting shell (sbatch passes the environment through).
#   sbatch --comment='E210 stage b' cluster/crosscoder_replicate_b.sbatch      # from the clone root
# Writes results/stage_b.json and results/stage_b_latents.csv. Estimated ~15 min, peak ~31 GB GPU, not measured.
#SBATCH --job-name=commute-cc-replicate-b
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
# l40-worker /ephemeral was 100% full on 2026-11-18 (job 61426): caches for this job go on the shared NFS instead
export HF_HOME=$HOME/commute-data/cache/hf TMPDIR=$HOME/commute-data/cache/tmp; mkdir -p "$HF_HOME" "$TMPDIR"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -n "${HF_TOKEN:-}" || { echo "STOP: HF_TOKEN is unset: export a token whose account accepted the Gemma-2 licence, then resubmit"; exit 1; }
.venv/bin/python experiments/crosscoder_replicate/run.py b
