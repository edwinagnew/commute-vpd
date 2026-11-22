#!/bin/bash
# P2.59 adl. Usage: sbatch --comment='E### stage0' cluster/adl.sbatch stage0 [0/1] [amend1]
#                   sbatch --comment='E### items' cluster/adl.sbatch items 0/1 [amend1]
# The third argument names an amendment folder: results/<amend>/ holds its predictions, outputs and log.
#SBATCH --job-name=commute-adl
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
STAGE="$1"; PART="${2:-0/1}"; AMEND="${3:-}"
OUT=experiments/adl/results${AMEND:+/$AMEND}
test -f "$OUT/predictions.json" || { echo "no $OUT/predictions.json"; exit 1; }
LOG=$OUT/run_${STAGE}_${PART/\//-}.log
( .venv/bin/python experiments/adl/run.py "$STAGE" "$PART" ${AMEND:+--amend "$AMEND"}; echo "=== FINISHED (exit $?) — results in $COMMUTE_REPO/$OUT ===" ) 2>&1 | tee -a "$LOG"
