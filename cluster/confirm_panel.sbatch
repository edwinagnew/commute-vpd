#!/bin/bash
# E276 (cell P2.66): the confirmatory text panel, train stage (GPU). Plan reports/plans/2026-11-23_plan_confirm-panel.md.
# Rules: experiments/confirm_panel/results/predictions.json (run.py refuses unless it is committed unchanged; train also
# refuses until the raters' tiers are filed in it). select, read and score run on the CPU (fleet/cpu-run, run.py header).
#   cd <clone> && ~/commute-agents/fleet/link-data <role or .>     (results/large must be the shared symlink)
#   for T in italian portuguese dutch indonesian turkish vietnamese welsh basque pubmed stackexchange books math; do
#     sbatch --comment="E276 stage: train $T" cluster/confirm_panel.sbatch train $T; done
# Expected on an L40, as E272's LM tasks (jobs 63727-63731: 3.0-4.8 min each; the job prints its seconds): ~5-10 min a
# task with the first HF stream; ~57 MB a state in bf16 under experiments/confirm_panel/results/large/states.
#SBATCH --job-name=commute-confirm-panel
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
.venv/bin/python experiments/confirm_panel/run.py "$@"
