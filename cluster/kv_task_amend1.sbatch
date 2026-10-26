#!/bin/bash
# kv_task Amendment 1: the Pile harness on E10's saved Adam 1e-5 states, no training
# (reports/plans/2026-10-23_plan_kv-task-amend1.md). About 18 min (shard 0) and 15 min (shard 1) on an L40.
#   cd ~/commute-agents/<role>
#   sbatch cluster/kv_task_amend1.sbatch --shard 0      # rand32: W0, the control rand32@30 first, then 100, 300, 10
#   sbatch cluster/kv_task_amend1.sbatch --shard 1      # ident32 100, 300, 10; once shard 0's log shows "[control] ... PASS"
# Needs on NFS: experiments/kv_task/results/{kv_task.json,predictions_amend1.json},
# experiments/kv_task/results/large/windows_{rand32,ident32}.pt and <task>/pullback/state_step{10,30,100,300}.pt (E10),
# experiments/transport_test/results/large/{alive,windows}.pt, ../windows.json. Before that: `cluster/sync.sh up`.
#SBATCH --job-name=commute-kv-amend1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
trap 'echo "=== FINISHED (exit $?) — results in $COMMUTE_REPO/experiments/kv_task/results/kv_task_amend1_shard{0,1}.json ==="' EXIT
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/kv_task/results/predictions_amend1.json || { echo "file the predictions first (results/predictions_amend1.json)"; exit 1; }
.venv/bin/python experiments/kv_task/amend1.py "$@"
