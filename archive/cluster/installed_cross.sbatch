#!/bin/bash
# E271 (cell P2.65): the installed-skill cross, controls stage then decomposition-free readings, on cluster debug ($0).
# Settings and bars: experiments/installed_cross/results/predictions.json (run.py refuses unless committed unchanged).
# PRED_SHA: taken from the environment if set, else from git here (the committed file's sha256). Submit from the clone root:
#   sbatch --comment='E271 rehearse: ctl' cluster/installed_cross.sbatch ctl
#   sbatch --comment='E271 rehearse: install' --dependency=afterok:<ctl> cluster/installed_cross.sbatch install addition   (cipher, yoruba alike, in parallel)
#   sbatch --comment='E271 rehearse: cross' --dependency=afterok:<3 installs> cluster/installed_cross.sbatch cross BASE TASK
#          (12 jobs: BASE in addition cipher yoruba M, TASK in addition cipher yoruba; in parallel)
#   ~/commute-agents/fleet/cpu-run python3 experiments/installed_cross/run.py controls   (no GPU; exit 1 = a control missed, stop there)
#   sbatch --comment='E271 stage: read' cluster/installed_cross.sbatch read BASE TASK   (12 jobs, as cross)
# Every job resumes from what it wrote: a job cut by the time limit is resubmitted with the same arguments.
# Times, from measured items (cluster L40, 2026-11-22): model load ~0.5 min (ctl 63568: 41 s wall all told, peak GPU 9.5 GB,
# host 6.1 GB); install 0.17 s per step with a lock read every 100 steps (install 63569: 2000 steps in 337 s), so 3 weights
# x 2000 steps = ~17 min plus up to 3 of M's own fine-tunes for the KL bar (<= 3000 steps each, the step time is
# unmeasured, ~0.1 s with the every-step stop read: <= 5 min each): it can pass 30 min; resubmit and it resumes per weight
# and per bar. cross per cell: 3 seeds x <= 3000 steps x ~0.1 s = <= 15 min. read per cell: 9 reads x ~25 s (E256's
# ~23 s per state at 256 rows) + ADL = ~5 min.
# Inputs: artifacts/vpd/ (the model); artifacts/sets/{val_0_64,val_0_600 or val_0_8192,val_0_8192,train_0_4096}.npy;
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
if [[ -z "${PRED_SHA:-}" || "$PRED_SHA" == SET_ME ]]; then
  PRED_SHA=$(git show HEAD:experiments/installed_cross/results/predictions.json | sha256sum | cut -d' ' -f1)
fi
export PRED_SHA
echo "PRED_SHA=$PRED_SHA"
.venv/bin/python experiments/installed_cross/run.py "$@"
