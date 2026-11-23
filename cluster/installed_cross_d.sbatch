#!/bin/bash
# E271 (cell P2.65) Stage D on cluster debug: the rehearsal on one base, or any stage_d.py job that fits 30 min.
# Two filed files: PRED_SHA (results/predictions.json) and PRED_SHA_D (results/predictions_stage_d.json), each taken from
# the environment if set, else from git here. Needs results/controls.json passed (run.py's controls).
#   sbatch --comment='E271 rehearse: stage D ctl + teach addition + one eb item' cluster/installed_cross_d.sbatch rehearse
# The rehearsal runs ctl (W0 under the Pile harness, seeds 0-2; task-row floors), teach addition (1000 student steps,
# F1-F3, 27 shares) and one E_b item (addition__addition_s0_size) at full read size. It measures: seconds per student
# step, the harness's seconds, seconds per share, the K scan's cases per second and seconds per item, peak GPU memory.
# Estimate (~20 min): model load 0.5 min; ctl ~4 min (6 harness blocks + 9 task-row adversaries); student ~2.5 min (E119's
# 1000 steps 109-167 s on an L40); faithfulness ~2 min; 27 shares ~2.5 min; one K scan ~8 min (E268's FLOP estimate).
#SBATCH --job-name=commute-installed-d
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
export HF_HOME=$HOME/commute-data/cache/hf TMPDIR=$HOME/commute-data/cache/tmp; mkdir -p "$HF_HOME" "$TMPDIR"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
sha_of() { git show "HEAD:experiments/installed_cross/results/$1" | sha256sum | cut -d' ' -f1; }
[[ -n "${PRED_SHA:-}" && "$PRED_SHA" != SET_ME ]] || PRED_SHA=$(sha_of predictions.json)
[[ -n "${PRED_SHA_D:-}" && "$PRED_SHA_D" != SET_ME ]] || PRED_SHA_D=$(sha_of predictions_stage_d.json)
export PRED_SHA PRED_SHA_D
echo "PRED_SHA=$PRED_SHA PRED_SHA_D=$PRED_SHA_D"
.venv/bin/python experiments/installed_cross/stage_d.py "$@"
