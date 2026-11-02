#!/bin/bash
# E55 Stage C2: the full decomposition on the cluster's normal QoS (user, 2026-11-02; decisions.md). Plan:
# reports/plans/2026-10-31_plan_e55-c2-vpd.md, ~56 L40-h as about three of these jobs. read and score stay on debug
# (cluster/pcfg_c2.sbatch).
#   cd ~/commute-agents/<role>
#   sbatch --job-name=commute-e55-c2 --dependency=singleton cluster/pcfg_c2_normal.sbatch
# Stage decompose of experiments/pcfg_anchors/c2.py, with a 1420-min deadline inside the 24-h limit. Exit 3 = saved, not
# finished: resubmit the same command by hand (compute nodes have no sbatch); it resumes from
# results/large/c2/decomp_train.pt. Exit 1 = an early check missed (saved), a non-finite loss (not saved) or a crash: the
# log says which. Exit 2 while predictions_c2.json is a draft or its early checks cannot work, or on a resume under another
# config or decompose.py. Exit 0 = results/large/c2/decomp_final.pt.
#SBATCH --job-name=commute-e55-c2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; puts this clone's src first on PYTHONPATH
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=decompose" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
out=experiments/pcfg_anchors/results/large/c2
mkdir -p "$out"
rc=0
.venv/bin/python experiments/pcfg_anchors/c2.py decompose --deadline-min 1420 2>&1 | tee -a "$out/c2_decompose.log" || rc=$?
if [ "$rc" = 3 ]; then
  echo "=== DEADLINE (exit 3): saved, not finished; resume by hand: sbatch --job-name=$SLURM_JOB_NAME --dependency=singleton cluster/pcfg_c2_normal.sbatch ==="
  exit 3
fi
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/pcfg_anchors/results/ ==="
exit $rc
