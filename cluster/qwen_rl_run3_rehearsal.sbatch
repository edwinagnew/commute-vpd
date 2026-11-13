#!/bin/bash
# P3.4 run 3, gate 3's $0 rehearsal on one cluster L40 (debug QoS): both null pods at E69's tiny size, then the read
# stage on the matched tiny state and the verdict. Plan: reports/plans/2026-11-13_plan_qwen-rl-readable-run3-gates.md,
# "Gate 3". Submit from the clone root after predictions_run3.json is committed (the run refuses otherwise):
#   sbatch --comment=E178 cluster/qwen_rl_run3_rehearsal.sbatch      # ~30 min
# 1. N_s70 at tiny_collapse: a forced collapse rule (every step under) ends the pod at step 2, no state saved.
# 2. N_s0 at tiny: pauses after step 1's checkpoint (exit 3), resumes from it (exit 0), its first read (step 2) meets a
#    target of 0, so it saves δW there and ends "matched".
# 3. read3.py: the controls at the tiny size (W0, E69's R step 100, 8 GSM8K problems), the matched state, the verdict
#    (seed 70 collapsed: "no call").
# States, checkpoints and GSM8K answers go to ~/commute-data/scratch/qwen-rl-lead/run3_tiny_<job> on /mnt/nw (a 17 GB
# checkpoint and a 5.64 GB δW: run3.py refuses under ~28 GB free), removed at the end; JSON and logs go to
# experiments/qwen_rl_readable/results/run3_tiny/ (earlier files there are moved to <name>.prev.<job>).
#SBATCH --job-name=commute-qwen_rl_run3_rehearsal
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
L=$HOME/commute-data/scratch/qwen-rl-lead/run3_tiny_$SLURM_JOB_ID
RES=experiments/qwen_rl_readable/results/run3_tiny
mkdir -p "$L" "$RES"
df -h "$L" | tail -1
trap 'rm -rf "$L"' EXIT
for f in "$RES"/*.json "$RES"/*.log "$RES/states"; do if [[ -e $f ]]; then mv "$f" "$f.prev.$SLURM_JOB_ID"; fi; done
PY=.venv/bin/python
RUN3="$PY experiments/qwen_rl_readable/run3.py --large $L"
READ3="$PY experiments/qwen_rl_readable/read3.py --size tiny --large $L"

expect() {  # expect <exit code> <command...>: run it, stop the job on any other exit
  local want=$1 got=0
  shift
  echo "=== $(date -u +%T) $* (expect exit $want)"
  "$@" || got=$?
  if [[ $got -ne $want ]]; then echo "=== REHEARSAL FAILED: exit $got from: $*"; exit 1; fi
}

expect 0 $RUN3 --pod N_s70 --size tiny_collapse   # COLLAPSED at step 2
expect 3 $RUN3 --pod N_s0 --size tiny             # paused after step 1's checkpoint
expect 0 $RUN3 --pod N_s0 --size tiny             # resumed; MATCHED at step 2, δW saved
du -sh "$L"/*
expect 0 $READ3 --stage gpu --controls-only
expect 0 $READ3 --stage gpu --states N_s0@2
expect 0 $READ3 --stage verdict
echo "=== REHEARSAL DONE"
