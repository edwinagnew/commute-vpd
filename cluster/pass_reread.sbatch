#!/bin/bash
# E48 pass re-read, one job per call. Plan: reports/plans/2026-10-29_plan_e48-pass-reread.md.
#   cd ~/commute-agents/<role>
#   sbatch --job-name=commute-e48-saved cluster/pass_reread.sbatch saved   # job 1: S1 control (stops on fallback), S2, S3; ~15 min
#   sbatch --job-name=commute-e48-regen cluster/pass_reread.sbatch regen   # job 2, only after job 1 read S1 'read'; ~20 min
# Before: experiments/pass_reread/results/predictions.json filed. Inputs on NFS: teacher_training/results/large/ (saved
# copies), teacher_training/results/*.json (filed readings), kv_task, task_ladder, transport_test results/large (windows,
# alive masks). Outputs: experiments/pass_reread/results/<job>.json after every record; regen also writes bf16 offsets
# to experiments/pass_reread/results/large/regen/ (6 x 60 MB). A rerun on the same card resumes the file.
#SBATCH --job-name=commute-e48
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
[ $# -eq 1 ] || { echo "pass_reread.sbatch: give exactly one job, saved or regen" >&2; exit 1; }
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/pass_reread/results/predictions.json || { echo "file the predictions first" >&2; exit 1; }
# results/large must be the shared symlink (fleet/link-data), never a real folder in the clone
[ -L experiments/pass_reread/results/large ] || ~/commute-agents/fleet/link-data "$COMMUTE_REPO"
[ -L experiments/pass_reread/results/large ] || { echo "experiments/pass_reread/results/large is not a symlink" >&2; exit 1; }
rc=0
.venv/bin/python experiments/pass_reread/run.py "$1" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/pass_reread/results/$1.json ==="
exit $rc
