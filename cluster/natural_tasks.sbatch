#!/bin/bash
# E124, the pronoun task, one stage per job, in order, each only after the last exited 0 (3: resubmit the same stage):
#   0  stage0.py, alone, W0 only: windows (kept counts per gender asserted before the model loads), E54 elicit s0 r and
#      the zero update (apparatus), W0 accuracy and headroom, P0 and l*, W0's removals. Exit 1 on any stop.
#   1  run.py --stage 1: six fine-tunes, each saved in bf16 to results/large/ as it is scored. Exit 3: resubmit.
#   2  run.py --stage 2: the random direction on every state first, P1-P3 and verdicts; pruning, reverse beside.
#   3  stage3.py: r and the stand-in ratio per state, then the filed scoring. Exit 3: resubmit.
# Plan: reports/plans/2026-11-06_plan_e124-natural-definition-tasks.md. Rules: experiments/natural_tasks/results/
# predictions.json and E112's experiments/case_label/results/predictions.json (both must be committed unchanged).
#   cd ~/commute-agents/<role> && ~/commute-agents/fleet/link-data <role> && sbatch cluster/natural_tasks.sbatch 0
# Needs cluster/natural_tasks_cache.sbatch's artifacts/sets/train_65536_524288.npy (job 54712; sha256 filed).
# Expected time on a debug L40, from E112's measured logs (experiments/case_label/results/stage*.out; 41-token windows,
# here 21): stage 0 E112 took 27 s after load; here + 2.1 GB rows, the name list, the E54 route read and 3 x (route
# cells + zero update) on 512 windows (E62: 6 route reads ~3.5 min): ~5-8 min. Stage 1 E112: 4 states in 35 s
# (up to 300 steps); here 6 states, up to 1,000 steps: <= 5 min. Stage 2 E112: ~1.5 min for 4 states; here 6: ~3 min.
# Stage 3: 6 states x (1 + subsets) = 14 route reads on <= 512 windows: ~8-12 min.
# Memory: E112 peaked at 7.8 GB host; stage 0 adds the two 2.1 GB row files and match chunks (dry run ran in 16G): 24G.
#SBATCH --job-name=commute-e124
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
D=experiments/natural_tasks
case "$1" in
  0) cmd=("$D/stage0.py") ;;
  1|2) cmd=("$D/run.py" --stage "$1") ;;
  3) cmd=("$D/stage3.py") ;;
  *) echo "stage must be 0-3"; exit 2 ;;
esac
rc=0
.venv/bin/python "${cmd[@]}" 2>&1 | tee -a "$D/results/stage$1.out" || rc=$?
echo "=== FINISHED E124 stage $1 (exit $rc) — results in $COMMUTE_REPO/$D/results/ ==="
exit $rc
