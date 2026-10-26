#!/bin/bash
# adjoint_sgd (E31) amendment 3: the unscaled adjoint split beside the scaled one. Plan: reports/plans/2026-10-22_plan_adjoint-sgd.md
# -> "Amendment 3". One emoticon seed-0 fine-tune per job, ~20 min (W0 ~6 min, the 1000-step fine-tune, six harness readings):
#   cd ~/commute-agents/<role>
#   sbatch --export=ALL,CELL=sgd cluster/adjoint_sgd_amend3.sbatch     # FIRST, alone
#   once its log shows G0 ([W0] ... "pass": true) and "[gate A3_G1 ...]" "ok": true: CELL=small, CELL=big
#   afterwards (laptop): experiments/adjoint_sgd/amend3.py --merge
# Before: `cluster/sync.sh up`; results/predictions_amend3.json filed. Inputs on NFS: experiments/transport_test/results/
# {windows.json, large/alive.pt, large/windows.pt}.
#SBATCH --job-name=commute-adjoint-sgd-a3
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
: "${CELL:?set CELL=sgd|small|big via --export}"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS cell=$CELL" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
for f in experiments/adjoint_sgd/results/predictions_amend3.json experiments/adjoint_sgd/results/adjoint_sgd.json \
         experiments/transport_test/results/windows.json experiments/transport_test/results/large/alive.pt \
         experiments/transport_test/results/large/windows.pt; do
  test -f "$f" || { echo "missing $f"; exit 1; }
done
.venv/bin/python experiments/adjoint_sgd/amend3.py "$CELL"
