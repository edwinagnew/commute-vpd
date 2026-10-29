#!/bin/bash
# E49 row blocks, one stage per job. Plan: reports/plans/2026-10-29_plan_e49-row-blocks.md (Cost and routing).
#   cd ~/commute-agents/<role>
#   sbatch --job-name=commute-e49-regen cluster/row_blocks.sbatch regen    # first, alone
#   sbatch --job-name=commute-e49-a cluster/row_blocks.sbatch a            # after regen
#   sbatch --job-name=commute-e49-b512s0 cluster/row_blocks.sbatch b512s0  # after a's controls: b512s{0,1,2}, b768s{0,1,2}
# --job-name names the log commute-e49-<stage>-<id>.out; without it the job is renamed at start, its log keeps %x.
# Before: experiments/row_blocks/results/predictions.json filed and committed. Inputs on NFS: broad_fragility's saved
# states (results/large), transport_test's alive.pt and windows.pt. Outputs: experiments/row_blocks/results/<stage>.json,
# results/large/<stage>.log (and dW_{adam,sgd,lora}.pt from regen); a rerun on the same card resumes the file.
# The score is CPU only: ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/row_blocks/run.py score
#SBATCH --job-name=commute-e49
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
[ $# -eq 1 ] || { echo "row_blocks.sbatch: give exactly one stage" >&2; exit 1; }
if [ "$SLURM_JOB_NAME" != "commute-e49-$1" ]; then   # submitted without --job-name: rename (the log keeps its name)
  scontrol update JobId="$SLURM_JOB_ID" JobName="commute-e49-$1" || true
  echo "renamed to commute-e49-$1; this log stays cluster/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
fi
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/row_blocks/results/predictions.json || { echo "file the predictions first" >&2; exit 1; }
rc=0
.venv/bin/python experiments/row_blocks/run.py "$1" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/row_blocks/results/$1.json ==="
exit $rc
