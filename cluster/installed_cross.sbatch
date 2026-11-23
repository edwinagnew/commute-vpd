#!/bin/bash
# E271 (cell P2.65): the installed-skill cross, controls stage then decomposition-free readings, on cluster debug ($0).
# Settings and bars: experiments/installed_cross/results/predictions.json (run.py refuses unless committed unchanged;
# PRED_SHA = `git show HEAD:experiments/installed_cross/results/predictions.json | sha256sum`). Submit from the clone root:
#   PRED_SHA=SET_ME sbatch --comment='E271 rehearse: ctl' cluster/installed_cross.sbatch ctl
#   PRED_SHA=SET_ME sbatch --comment='E271 rehearse: install' --dependency=afterok:<ctl> cluster/installed_cross.sbatch install addition   (cipher, yoruba alike, in parallel)
#   PRED_SHA=SET_ME sbatch --comment='E271 rehearse: cross' --dependency=afterok:<3 installs> cluster/installed_cross.sbatch cross addition (cipher, yoruba, M alike, in parallel)
#   ~/commute-agents/fleet/cpu-run python3 experiments/installed_cross/run.py controls   (no GPU; exit 1 = a control missed, stop there)
#   PRED_SHA=SET_ME sbatch --comment='E271 stage: read' cluster/installed_cross.sbatch read addition   (cipher, yoruba, M alike)
# Times (unmeasured; the rehearsal measures them, estimated from E256's measured items): model load ~2 min; ctl ~4 min
# (Yoruba streaming + 3 x 512 reads); install ~3 x 2000 steps at ~30 ms = ~3 min + 1-3 M fine-tunes; cross per base 9
# fine-tunes at <= 3000 steps x ~25 ms (a step + a 256-row read) = <= 11 min; read per base 27 reads x ~25 s (E256's
# ~23 s per state at 256 rows, stage-2 logs) + ADL ~5 s = ~12 min. Each job is under the 30 min debug cap.
# Inputs: artifacts/vpd/ (the model); artifacts/sets/{val_0_64,val_64_512 (cut from val_0_8192),val_0_8192,train_0_4096}.npy;
# experiments/realistic_tasks/results/predictions_run2.json, experiments/share_untrained/results/stage0.json (k_m),
# experiments/neuron_r/results/predictions.json (k_half settings); Yoruba Wikipedia streamed from Hugging Face in ctl.
#SBATCH --job-name=commute-installed
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
[[ "${PRED_SHA:-SET_ME}" != SET_ME ]] || { echo "set PRED_SHA to the committed predictions.json's sha256"; exit 1; }
.venv/bin/python experiments/installed_cross/run.py "$@"
