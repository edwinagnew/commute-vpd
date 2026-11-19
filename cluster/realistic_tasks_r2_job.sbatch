#!/bin/bash
# E211 (cell P2.38 run 2): every scorecard method on the real-text pairs run 1 kept. Plan
# reports/plans/2026-11-18_plan_p2-38-run2.md. Rules: experiments/realistic_tasks/results/predictions_run2.json (run2.py
# refuses unless it is committed unchanged). One job per argument: ctl first and alone, then the six tasks after it:
#   cd <clone> && ~/commute-agents/fleet/link-data <role or .>
#   sbatch --comment=E211 cluster/realistic_tasks_r2_job.sbatch ctl
#   for each of L1 L3 T1 T2 T3 T4: sbatch --comment='E211 stage: <task>' --dependency=afterok:<ctl id> cluster/realistic_tasks_r2_job.sbatch <task>
# Each job writes experiments/realistic_tasks/results/run2_<job>.json as every state lands, prints "STAGE <job> DONE" at
# the end; a filed stop prints "STOP: ..." and exits 1. Task jobs save each update in bf16 under
# experiments/realistic_tasks/results/large/run2/ (about 57 MB each, 6 a task).
# Inputs: E53's two saved states and E136/E143/E98/update_svd records (ctl), cached Pile rows (artifacts/sets
# train_0_4096, val_0_256), the model in artifacts/vpd/; downloads inside the job: AG News (19 MB), SST-2 (3 MB, run 1's
# builder loads it), Wikipedia fr/sw/tl/yo streamed (first ~1.3M tokens each).
# Expected on an L40 (run 1's log 61424; E172's k_half log 59606): ctl ~6 min; T1-T4 ~21 min each (13 full reads of 500
# windows of 127 tokens at ~85 s); L1, L3 ~6 min each. 45-min limit (debug allows 2 h).
#SBATCH --job-name=commute-realistic-r2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
export HF_HOME=$HOME/commute-data/cache/hf TMPDIR=$HOME/commute-data/cache/tmp; mkdir -p "$HF_HOME" "$TMPDIR"  # l40-worker's /ephemeral was full (2026-11-18)
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/realistic_tasks/results/predictions_run2.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/realistic_tasks/run2.py "$1"
