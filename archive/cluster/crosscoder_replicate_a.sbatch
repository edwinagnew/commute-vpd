#!/bin/bash
# E210 (cell P2.37) stage a, CPU: our relative norm on Minder et al.'s released Gemma-2-2B crosscoder against their
# feature_df dec_norm_diff. Plan: reports/plans/2026-11-18_plan_p2-37-crosscoder-replicate.md.
#   sbatch --comment=E210 cluster/crosscoder_replicate_a.sbatch      # from the clone root
# Writes experiments/crosscoder_replicate/results/stage_a.json; a miss logs "STOP: ..." and exits 1.
# Downloads 2.72 GB + 41.5 MB into HF_HOME (/ephemeral). ~5 min, ~8 GB host RAM.
#SBATCH --job-name=commute-cc-replicate-a
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1   # stage a loads no model, but the no-compute hook reads run.py as a whole
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/crosscoder_replicate/run.py a
