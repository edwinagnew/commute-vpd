#!/bin/bash
# Many cues, one answer (E33). Plan: reports/plans/2026-10-22_plan_many-cues.md.
#   cd ~/commute-agents/<role>
#   sbatch cluster/many_cues.sbatch     # ~5-8 min on an L40: G1 (emoticon vs E28) first, stops the job on a fail
# Before that: `cluster/sync.sh up` and predictions filed (results/predictions.json). Gitignored inputs on NFS:
# experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt};
# experiments/task_ladder/results/large/windows_r1_t588_v452.pt; experiments/regating_tasks/results/regating_tasks.json
# (in git). The mining stage reads Pile train rows 0..32768 (data.pile_rows; cached under artifacts/sets by E5).
#SBATCH --job-name=commute-many-cues
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/many_cues/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
rc=0; .venv/bin/python experiments/many_cues/run.py "$@" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/many_cues/results/ (many_cues.json, large/) ==="
exit $rc
