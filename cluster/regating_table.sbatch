#!/bin/bash
# E46 stage 0, the re-gating table, and its amendments. Plan: reports/plans/2026-10-28_plan_regating-line-reset.md ("Amendment 1 adopted").
#   cd ~/commute-agents/<role>
#   sbatch cluster/regating_table.sbatch 0a   # the control alone: W0 cells, identity, cached and rebuilt states; exit 1 on a fail
#   sbatch cluster/regating_table.sbatch 0b   # only after results/stage0a.json passed: the table and the scored verdicts
#   sbatch cluster/regating_table.sbatch amend1  # Amendment 2: the seven random rows of 0b, scale cap 100 (~3-5 min)
#   sbatch cluster/regating_table.sbatch amend3  # Amendment 3: the ratio on split nulls, E12 tables (~12-16 min)
# Before that: predictions filed (experiments/regating_table/results/predictions.json). Inputs on NFS (gitignored):
# experiments/recruitment_test/results/large/states/ (E38's nine E18 states), experiments/reading_a_coverage/results/large/
# emoticon/dW_ft_step10_s0.pt, experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt},
# experiments/task_ladder/results/large/windows_{r1_t534_v369,r2_t359_v816}.pt, Pile rows artifacts/sets/train_0_32768.npy
# and val_0_8192.npy (streamed from Hugging Face on first use by 0a). 0a writes E12's windows and weight changes and
# E36's (~1.2 GB, fp32) to experiments/regating_table/results/large/; 0b reads them.
#SBATCH --job-name=commute-regating-table
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# The shared .venv's editable install points at whichever clone last ran `uv sync`: put this clone's src first (as
# recruitment_test.sbatch). TEMPORARY: remove once the shared venv no longer pins one clone.
export PYTHONPATH="$COMMUTE_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
test $# -eq 1 || { echo "usage: sbatch cluster/regating_table.sbatch 0a|0b|amend1|amend3"; exit 2; }
test -f experiments/regating_table/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
case "$1" in
  0a|0b) .venv/bin/python experiments/regating_table/run.py --stages "$1" ;;
  amend1) .venv/bin/python experiments/regating_table/amend1.py ;;  # refuses without results/predictions_amend1.json
  amend3) .venv/bin/python experiments/regating_table/amend3.py ;;  # refuses without results/predictions_amend3.json
  *) echo "usage: sbatch cluster/regating_table.sbatch 0a|0b|amend1|amend3"; exit 2 ;;
esac
