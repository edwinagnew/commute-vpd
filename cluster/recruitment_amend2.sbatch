#!/bin/bash
# E38 Amendment 2 (trigger-only and everywhere removal from both models; census counters). Plan: reports/plans/2026-10-27_plan_e38-amend2.md.
#   cd ~/commute-agents/<role>
#   sbatch cluster/recruitment_amend2.sbatch --stages 0     # the controls C0-C4 alone; exit 1 on a fail
#   sbatch cluster/recruitment_amend2.sbatch --stages 1,2   # only after amend2_stage0.json passed
# Arguments pass through to experiments/recruitment_test/run.py. Before that: predictions filed
# (experiments/recruitment_test/results/predictions.json). Inputs on NFS (gitignored):
# experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt},
# experiments/task_ladder/results/large/windows_{r1_t534_v369,r2_t359_v816}.pt,
# experiments/regating_leak/results/{regating_leak.json, large/<task>_s{0,1}_shift.pt}; Pile val rows 0..2000
# (artifacts/sets/val_0_2000.npy, streamed from Hugging Face on first use by stage 2). Stage 0 writes the nine weight
# changes (~1 GB, fp32) to experiments/recruitment_test/results/large/states/; stages 1-3 read them.
#SBATCH --job-name=commute-recruitment-amend2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# The shared .venv's editable install (commute.pth) points at whichever clone last ran `uv sync` (commute-lead's on
# 2026-10-27), which lacks commute.recruitment, routing and census: put this clone's src first. TEMPORARY: remove once
# the shared venv no longer pins one clone.
export PYTHONPATH="$COMMUTE_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
test -f experiments/recruitment_test/results/predictions_amend2.json || { echo "file the predictions first (results/predictions_amend2.json)"; exit 1; }
.venv/bin/python experiments/recruitment_test/amend2.py "$@"
