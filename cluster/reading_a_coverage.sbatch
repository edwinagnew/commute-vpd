#!/bin/bash
# Reading-A coverage on one task (2026-10-20: the emoticon task, at the user's request). Plan:
# reports/2026-10-19_plan_reading-a-coverage.md; the five-task run itself went to RunPod on 2026-10-19.
#   cd /mnt/nw/home/e.agnew/commute
#   sbatch cluster/reading_a_coverage.sbatch --task emoticon            # scored, ~15 min on an L40
#   sbatch cluster/reading_a_coverage.sbatch --task emoticon --smoke    # smoke, ~5 min
# Before that: `cluster/sync.sh up` and predictions filed (results/predictions.json).
# Expected: W0 floors (3 adversary seeds, ~2 min each) + 2 scored Pile passes + two 10-step trainings: ~15 min, < 20 GB VRAM.
#SBATCH --job-name=commute-reading-a-coverage
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/nw/home/e.agnew/commute/cluster/logs/%x-%j.out

set -euo pipefail
source /mnt/nw/home/e.agnew/commute/cluster/env.sh
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/reading_a_coverage/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/reading_a_coverage/run.py "$@"
