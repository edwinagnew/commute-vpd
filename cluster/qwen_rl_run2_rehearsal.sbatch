#!/bin/bash
# P3.4 run 2, gate 3's $0 rehearsal on one cluster L40: every pod stage at E69's tiny size (4 problems x 2 samples a step,
# 2 steps, saves at 1 and 2, a pause at step 1 and a resume from its checkpoint), then, with R_s70, the chosen-steps helper
# and the cluster read stage (controls and one tiny saved state on 8 GSM8K problems; the outside share on the first 7
# matrices). Plan: reports/plans/2026-11-12_plan_qwen-rl-readable-run2-gates.md, "Rehearsal". Submit from the clone root,
# after predictions_run2.json is committed; three jobs side by side, one pod each (R_s70 also runs the read stage):
#   sbatch --comment=E### cluster/qwen_rl_run2_rehearsal.sbatch R_s70     # ~11 min of pod, ~8 of reads
#   sbatch --comment=E### cluster/qwen_rl_run2_rehearsal.sbatch N_s70     # ~11 min
#   sbatch --comment=E### cluster/qwen_rl_run2_rehearsal.sbatch F_s70     # ~12 min (samples W0 in the loop)
# States, checkpoints and GSM8K answers go to this node's /ephemeral/$USER/run2_tiny_<job> (one δW 5.64 GB, a checkpoint
# 17 GB and its .tmp: run2.py refuses under ~50 GB free), removed at the end; JSON and logs to
# experiments/qwen_rl_readable/results/run2_tiny/ (a pod's earlier files there are moved to <name>.prev.<job>).
# Each pod's log has its step times, checkpoint and save seconds, disk line and peaks: the numbers gate 3 measures.
# Host memory: run 1's pods peaked at 11.4 GB (results/*.json peaks); a δW adds 5.6 GB in the read stage.
#SBATCH --job-name=commute-qwen_rl_run2_rehearsal
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
L=/ephemeral/$USER/run2_tiny_$SLURM_JOB_ID
RES=experiments/qwen_rl_readable/results/run2_tiny
mkdir -p "$L" "$RES"
df -h "$L" | tail -1
trap 'rm -rf "$L"' EXIT
PY=.venv/bin/python
RUN2="$PY experiments/qwen_rl_readable/run2.py --size tiny --large $L"
READ2="$PY experiments/qwen_rl_readable/read2.py --size tiny --large $L"

expect() {  # expect <exit code> <command...>: run it, stop the job on any other exit
  local want=$1 got=0
  shift
  echo "=== $(date -u +%T) $* (expect exit $want)"
  "$@" || got=$?
  if [[ $got -ne $want ]]; then echo "=== REHEARSAL FAILED: exit $got from: $*"; exit 1; fi
}

for pod in "$@"; do
  for f in "$RES/$pod.json" "$RES/$pod.log"; do if [[ -e $f ]]; then mv "$f" "$f.prev.$SLURM_JOB_ID"; fi; done
  expect 3 $RUN2 --pod "$pod"   # pauses after step 1's checkpoint, save and read
  expect 0 $RUN2 --pod "$pod"   # resumes from the step-1 checkpoint, runs step 2
  du -sh "$L/$pod"
  [[ $pod == R_s70 ]] || rm -rf "$L/$pod/dW"   # keep R_s70's states for the read stage
done
if [[ " $* " == *" R_s70 "* ]]; then
  for f in controls.json states reads.log share_sites.jsonl share.json chosen.json; do if [[ -e $RES/$f ]]; then mv "$RES/$f" "$RES/$f.prev.$SLURM_JOB_ID"; fi; done
  expect 0 $RUN2 --chosen
  expect 0 $READ2 --stage gpu --controls-only
  expect 0 $READ2 --stage gpu --states R_s70@2
  expect 0 $READ2 --stage share --sites 7
  expect 0 $READ2 --stage verdict
fi
echo "=== REHEARSAL DONE"
