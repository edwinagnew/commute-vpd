#!/bin/bash
# E38 Amendment 4 (the change-matched restore: controls R0-R5 on all nine states, then C_n, C_m, all_alive and the
# emoticon L3 arms). Plan: reports/plans/2026-10-28_plan_e38-amend4.md.
#   cd ~/commute-agents/<role>
#   sbatch cluster/recruitment_amend4.sbatch     # one job; exit 1 on a control fail
# Arguments pass through to experiments/recruitment_test/amend4.py. Before that: predictions filed
# (experiments/recruitment_test/results/predictions_amend4.json and Amendment 3's) and Amendment 2's stage 0 passed
# (results/amend2_stage0.json). Inputs on NFS (gitignored): Amendment 2's nine weight changes in
# experiments/recruitment_test/results/large/states/ (sha256-checked against amend2_stage0.json);
# experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt},
# experiments/task_ladder/results/large/windows_{r1_t534_v369,r2_t359_v816}.pt. No fine-tuning, no Pile rows.
#SBATCH --job-name=commute-recruitment-amend4
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G  # job 40041 ran at 40G; 64G waited on node RAM (2026-10-28)
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# The shared .venv's editable install (commute.pth) points at whichever clone last ran `uv sync`: put this clone's src
# first. TEMPORARY: remove once the shared venv no longer pins one clone.
export PYTHONPATH="$COMMUTE_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
test -f experiments/recruitment_test/results/predictions_amend4.json || { echo "file the predictions first (results/predictions_amend4.json)"; exit 1; }
.venv/bin/python experiments/recruitment_test/amend4.py "$@"
