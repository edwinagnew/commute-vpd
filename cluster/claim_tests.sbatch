#!/bin/bash
# E47 claim tests, one stage per job. Plan: reports/plans/2026-10-28_plan_claim-tests.md (section 6, Amendment 1).
#   cd ~/commute-agents/<role>
#   sbatch --job-name=commute-e47-t1 cluster/claim_tests.sbatch t1     # stages t1 t1b t2 t2b t2c t3 t4 t4b t5 t5b
# --job-name names the log commute-e47-<stage>-<id>.out; without it the job is renamed at start, its log keeps %x.
# Before: experiments/claim_tests/results/predictions.json filed with the stage. Inputs on NFS (saved states under
# teacher_training/ and broad_fragility/ results/large/, windows under transport_test, task_ladder, kv_task). Outputs:
# experiments/claim_tests/results/<stage>.json, results/large/<stage>.log; a rerun on the same card resumes the file.
#SBATCH --job-name=commute-e47
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:35:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
[ $# -eq 1 ] || { echo "claim_tests.sbatch: give exactly one stage" >&2; exit 1; }
if [ "$SLURM_JOB_NAME" != "commute-e47-$1" ]; then   # submitted without --job-name: rename (the log keeps its name)
  scontrol update JobId="$SLURM_JOB_ID" JobName="commute-e47-$1" || true
  echo "renamed to commute-e47-$1; this log stays cluster/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
fi
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/claim_tests/results/predictions.json || { echo "file the predictions first" >&2; exit 1; }
rc=0
.venv/bin/python experiments/claim_tests/run.py "$1" || rc=$?
echo "=== sstat (peak host memory of this job's steps) ==="
sstat -j "${SLURM_JOB_ID}.batch" --format=JobID,MaxRSS,MaxVMSize,AveCPU -P || echo "sstat unavailable"
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/claim_tests/results/$1.json ==="
exit $rc
