#!/bin/bash
# E279 (cell P2.66): the untrained-SVD share on E272's wide KL-hi panel. Rules: experiments/untrained_panel/results/
# predictions.json (run.py refuses unless it is committed unchanged).
#   cd <clone> && ~/commute-agents/fleet/link-data <role or .>   (convergence_panel/results/large must be the shared symlink)
#   sbatch --comment=E279 cluster/untrained_panel.sbatch 0                                          (alone, first)
#   after it passes, three jobs (any order, together):
#     sbatch --comment=E279 cluster/untrained_panel.sbatch read arxiv category conll french
#     sbatch --comment=E279 cluster/untrained_panel.sbatch read german lower sentiment spanish
#     sbatch --comment=E279 cluster/untrained_panel.sbatch read squad summary swahili yoruba
#   ~/commute-agents/fleet/cpu-run python3 experiments/untrained_panel/run.py score; then plot.py
# Expected on an L40: one untrained read (unmasked forward with gates + one masked forward over 500 rows) <= ~4.4 s
# (E136 job 56944 read 68 states on SVD and rotation, set-up and checks included, in 5 min); up to ~2x on squad's
# 256-token prompts. Stage 0 ~3 min (model load, 2 readers, 5 reads); a task ~1-3 min (build, 5 reads); 4 tasks
# ~12 min. No file is saved beyond the JSONs.
#SBATCH --job-name=commute-untrained-panel
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
.venv/bin/python experiments/untrained_panel/run.py "$@"
