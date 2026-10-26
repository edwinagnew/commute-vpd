#!/bin/bash
# adjoint_sgd (E31) amendment 1: task vs SGD rate. Plan: reports/plans/2026-10-22_plan_adjoint-sgd.md -> "Amendment 1".
# One cell per job, each ~22 min on an L40 (W0 ~6 min, the 10-step gate, one 1000-step fine-tune read at 10/100/1000):
#   cd ~/commute-agents/<role>
#   sbatch --export=ALL,CELL=r1 cluster/adjoint_sgd_amend1.sbatch    # FIRST, alone
#   once its log shows G0 ([W0] ... "pass": true) and "[gate A1_G1 r1]" ok: CELL=r0 and CELL=emo
#   afterwards (laptop): run.py amend1_merge
# Before: `cluster/sync.sh up`; results/predictions_amend1.json filed. Inputs on NFS: experiments/transport_test/results/
# {windows.json, large/alive.pt, large/windows.pt}, experiments/task_ladder/results/large/windows_{r0_t352_v310,r1_t534_v369}.pt.
#SBATCH --job-name=commute-adjoint-sgd-a1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
: "${CELL:?set CELL=r1|r0|emo via --export}"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS cell=$CELL" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
for f in experiments/adjoint_sgd/results/predictions_amend1.json experiments/transport_test/results/windows.json \
         experiments/transport_test/results/large/alive.pt experiments/transport_test/results/large/windows.pt \
         experiments/task_ladder/results/large/windows_r0_t352_v310.pt experiments/task_ladder/results/large/windows_r1_t534_v369.pt; do
  test -f "$f" || { echo "missing $f"; exit 1; }
done
.venv/bin/python experiments/adjoint_sgd/run.py amend1 "$CELL"
