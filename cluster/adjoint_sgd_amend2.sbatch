#!/bin/bash
# adjoint_sgd (E31) amendment 2: seed 1 of the five single-seed cells. Plan: reports/plans/2026-10-22_plan_adjoint-sgd.md ->
# "Amendment 2". One cell per job, each ~17-22 min (W0 ~6 min, the 10-step seed-0 control, one 1000-step seed-1
# fine-tune read at 10/100/1000):
#   cd ~/commute-agents/<role>
#   sbatch --export=ALL,CELL=r1_01 cluster/adjoint_sgd_amend2.sbatch    # FIRST, alone
#   once its log shows G0 ([W0] ... "pass": true) and "[gate A2_G1 r1_01]" ok: CELL=r0_01, r1_03, r0_03, emo_03
#   afterwards (laptop): run.py amend2_merge
# Before: `cluster/sync.sh up`; results/predictions_amend2.json filed. Inputs on NFS: experiments/transport_test/results/
# {windows.json, large/alive.pt, large/windows.pt}, experiments/task_ladder/results/large/windows_{r0_t352_v310,r1_t534_v369}.pt.
#SBATCH --job-name=commute-adjoint-sgd-a2
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
: "${CELL:?set CELL=r1_01|r0_01|r1_03|r0_03|emo_03 via --export}"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS cell=$CELL" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
for f in experiments/adjoint_sgd/results/predictions_amend2.json experiments/transport_test/results/windows.json \
         experiments/transport_test/results/large/alive.pt experiments/transport_test/results/large/windows.pt \
         experiments/task_ladder/results/large/windows_r0_t352_v310.pt experiments/task_ladder/results/large/windows_r1_t534_v369.pt; do
  test -f "$f" || { echo "missing $f"; exit 1; }
done
.venv/bin/python experiments/adjoint_sgd/run.py amend2 "$CELL"
