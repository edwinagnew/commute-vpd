#!/bin/bash
# Old-task labels (cell P1.7): the 17 old 67M tasks' saved fine-tunes read with E137's search as E139 reads a fine-tune, labelled by
# E139's filed rule (provisional). Plan: reports/plans/2026-11-10_plan_old-task-labels.md. Rules: experiments/old_task_labels/results/predictions.json
# (the run refuses unless it is committed unchanged). The lead replaces the placeholder experiment number below (and `EXPERIMENT` in run.py)
# when it claims the run. One L40, debug QoS, $0. Submit from the clone root:
#   sbatch cluster/old_task_labels.sbatch control                      (stage 0, alone and first: about 12 min; exit 1 if it fails;
#                                                                       needs one finished E139 item with a reading)
#   sbatch cluster/old_task_labels.sbatch copy24:0 copy24:1 rank1:0    (three items per job; several jobs side by side)
# Items are task:seed (tasks in commute.old_tasks.TASKS, seeds 0 and 1; rand1 seed 0 only). Exit 3: the wall limit (105 min) was reached
# with results saved: resubmit the same line, it resumes after the last finished search.
# Inputs: the saved delta-W files and task files under experiments/*/results/large (commute.old_tasks), E139's results and large/<item>.pt.
# Writes: results/control.json, results/<task>_s<seed>.json (no weights saved). Time on an L40, from E139's logs (job 57251): item about
# 34 min; memory about 35 GB of the card at the `final` site, host about 6 GB.
#SBATCH --job-name=commute-oldtasks
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:55:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --comment=E142

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/old_task_labels/run.py "$@"
