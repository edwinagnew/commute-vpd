#!/bin/bash
# E273 realtext_eb (cell P2.64). Rehearsal ($0, full read sizes: stage 0 then one label and one text item, each timed,
# peak GPU memory in its item JSON; a rerun skips written items, so a timed-out job resumes):
#   sbatch --comment='E273 rehearse: full sizes, stage 0 + L1_s0 + T1_s0' cluster/realtext_eb.sbatch items rh --only L1_s0 T1_s0
# Later shards (if the rehearsal's per-item time fits debug): cluster/realtext_eb.sbatch items c<I> --shard I N
#SBATCH --job-name=commute-realtext-eb
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/realtext_eb/results/predictions.json || { echo "no predictions"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
STAGE="$1"; TAG="$2"; shift 2
LOG=experiments/realtext_eb/results/run_${STAGE}_${TAG}.log
.venv/bin/python experiments/realtext_eb/run.py --stage "$STAGE" --tag "$TAG" "$@" 2>&1 | tee -a "$LOG"
CODE=${PIPESTATUS[0]}
echo "=== FINISHED (exit $CODE) — results in $COMMUTE_REPO/experiments/realtext_eb/results ===" | tee -a "$LOG"
exit $CODE
