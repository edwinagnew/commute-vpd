#!/bin/bash
# Cell P2.74 run 1 (E292): the old near-realistic 67M tasks read with every current reading. Rules and settings
# experiments/old_tasks_read/results/predictions.json (both run files refuse unless it is committed unchanged). The first
# argument picks the file: `run` (run.py: 0, read, train, read_new) or `defs` (run_defs.py: eb_neurons, eb_vpd, eapig).
# Submit from the clone root, after ~/commute-agents/fleet/link-data . (results/large symlinks):
#   sbatch --comment=E292 cluster/old_tasks_read.sbatch run 0                       (alone and first; exit 1 on a miss)
#   sbatch --comment=E292 cluster/old_tasks_read.sbatch run read TASK [TASK ...]
#   sbatch --comment=E292 cluster/old_tasks_read.sbatch run train TASK [TASK ...]
#   sbatch --comment=E292 cluster/old_tasks_read.sbatch run read_new TASK [TASK ...]
#   sbatch --comment=E292 cluster/old_tasks_read.sbatch defs eb_neurons --shard I N --tag nI
#   sbatch --comment=E292 cluster/old_tasks_read.sbatch defs eb_vpd --shard I N --tag vI
#   sbatch --comment=E292 cluster/old_tasks_read.sbatch defs eapig --tag e0       (needs the fc310 venv, as cluster/eapig_overlap.sbatch)
#SBATCH --job-name=commute-old-tasks-read
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:55:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
for d in old_tasks_read transport_test task_ladder recruitment_test reading_a_coverage many_cues performed_anchor case_label real_vs_twin share_untrained; do
  test -L experiments/$d/results/large || { echo "experiments/$d/results/large is not linked: run ~/commute-agents/fleet/link-data ."; exit 1; }
done
file=$1; shift
case "$file" in
  run) script=run.py ;;
  defs) script=run_defs.py ;;
  *) echo "first argument: run or defs"; exit 2 ;;
esac
if [[ "$file" == defs && "${1:-}" == eapig ]]; then
  test -x "$HOME/commute-data/venvs/fc310-fdaeeaa/bin/python" || { echo "no fc310 venv: sbatch cluster/eapig_overlap_venv.sbatch first"; exit 1; }
  set -- "$@" --scratch "/ephemeral/$USER/e292_eapig/$SLURM_JOB_ID"
fi
.venv/bin/python experiments/old_tasks_read/$script "$@"
