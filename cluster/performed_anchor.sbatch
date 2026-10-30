#!/bin/bash
# E53 performed anchor. Plan: reports/plans/2026-10-29_plan_performed-anchor.md. One stage per job, in order:
#   cd ~/commute-agents/<role>
#   sbatch cluster/performed_anchor.sbatch a0         # W0 only (~1 min): results/a0.json; exit 0 GO, 1 STOP, 2 error
#   sbatch cluster/performed_anchor.sbatch finetune   # the four fine-tunes (~3-5 min): results/finetune.json, large/dW_*.pt
#   sbatch cluster/performed_anchor.sbatch read       # controls, nulls, readings (~10-20 min): results/read.json
# Amendment 1 (plan section "Amendment 1 (2026-10-30)"; predictions results/predictions_amend1.json), after E53:
#   sbatch cluster/performed_anchor.sbatch a1_finetune  # W0 check (stop rule), four fine-tunes (~3-4 min):
#                                                       # results/check_amend1.json, finetune_amend1.json, large/dW_*.pt
#   sbatch cluster/performed_anchor.sbatch a1_read      # E53 repro control, 12 states, random nulls (~4-8 min):
#                                                       # results/read_amend1.json; needs E53's read.json, finetune.json
#                                                       # and large/dW_lowercase_s0_step300.pt
# Amendment 1b (narrow_lower stopped at the check; predictions results/predictions_amend1b.json):
#   sbatch cluster/performed_anchor.sbatch a1b_finetune # broad cipher x 2 seeds (~2 min): results/finetune_amend1b.json
#   sbatch cluster/performed_anchor.sbatch a1b_read     # E53 re-reads, 6 states, random null (~4-6 min):
#                                                       # results/read_amend1b.json; needs E53's four step-300 states
# Before: the stage's predictions filed and committed (a0: results/predictions_a0.json; finetune, read:
# results/predictions.json); finetune and read refuse unless a0.json is a GO; read refuses unless finetune.json is done.
# Inputs: the VPD artifacts, Pile rows artifacts/sets/{val_0_256,train_0_4096}.npy (cached),
# experiments/transport_test/results/large/alive.pt (read). Logs: results/large/<stage>.log. Exit 1 = a control missed.
# On a RunPod pod: the same command lines (`python experiments/performed_anchor/run.py finetune|read`) from the repo root
# with this file's environment (PYTHONPATH=<repo>/src, caches off /tmp).
#SBATCH --job-name=commute-e53
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
stage=${1:-}
case "$stage" in
  a0) script=experiments/performed_anchor/a0.py; pred=predictions_a0.json; args=() ;;
  finetune|read) script=experiments/performed_anchor/run.py; pred=predictions.json; args=("$stage") ;;
  a1_finetune|a1_read) script=experiments/performed_anchor/run.py; pred=predictions_amend1.json; args=("$stage") ;;
  a1b_finetune|a1b_read) script=experiments/performed_anchor/run.py; pred=predictions_amend1b.json; args=("$stage") ;;
  *) echo "usage: sbatch cluster/performed_anchor.sbatch a0|finetune|read|a1_finetune|a1_read|a1b_finetune|a1b_read" >&2; exit 2 ;;
esac
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# The shared .venv's editable install points at whichever clone last ran `uv sync`: put this clone's src first (as
# restore_anchors.sbatch). TEMPORARY: remove once the shared venv no longer pins one clone.
export PYTHONPATH="$COMMUTE_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
test -f "experiments/performed_anchor/results/$pred" || { echo "file the predictions first: $pred" >&2; exit 1; }
mkdir -p experiments/performed_anchor/results/large
rc=0
.venv/bin/python "$script" "${args[@]}" 2>&1 | tee "experiments/performed_anchor/results/large/$stage.out" || rc=$?
echo "=== FINISHED $stage (exit $rc) — results in $COMMUTE_REPO/experiments/performed_anchor/results/ ==="
exit $rc
