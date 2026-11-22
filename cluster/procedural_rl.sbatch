#!/bin/bash
# E_TBD_P (cell P2.56 run 1): three procedural tasks on the 67M by RL and by SFT, then E222's readings of every state.
# Settings, bars and expected values: experiments/procedural_rl/results/predictions.json (run.py refuses unless it is
# committed unchanged; set PRED_SHA=SET_ME to `git show HEAD:experiments/procedural_rl/results/predictions.json | sha256sum`).
# One job per argument list; submit from the clone root, ctl first and alone:
#   PRED_SHA=SET_ME sbatch --comment=E_TBD_P --time=00:20:00 --mem=40G cluster/procedural_rl.sbatch ctl
#   PRED_SHA=SET_ME sbatch --comment=E_TBD_P --dependency=afterok:<ctl id> cluster/procedural_rl.sbatch train sequence   (brackets, addition alike, in parallel; only the tasks stage0.json lists as usable)
#   PRED_SHA=SET_ME sbatch --comment=E_TBD_P --dependency=afterok:<train id> cluster/procedural_rl.sbatch read sequence    (each after its own train job)
# A task's train job exits 1 when RL seed 0 fails the learned gate: the task is dropped, its read job is not submitted.
# Times on an L40, each a measured item x its count: model load ~2 min (E218); ctl ~7 min (E211's ctl ~2.5 min, log
# commute-rl67m-s3-61974; items, zero update and benchmark ~2 min); train per task ~6 min (RL ~13 s per seed at 22-31
# steps, E224 stage2c_arm_2_rl_1_2.json; SFT at most ~40 s per seed, E220 stage2b_1_demo.json; 9 stop reads, KL reads and
# saves of ~57 MB); read per task ~8 min (reader ~1 min, E211; 10 reads (zero + 9 states) x ~23 s, E222's 18 s per state
# at 200 rows, stage3_ctl.json "zero seconds", scaled to 256). Wall clock with the three tasks in parallel ~21 min.
# Host memory: ctl reads E53's gate sets for E211's ctl (~20 GB, as E222's ctl at 40G); train and read under 10 GB.
# Inputs: the model in artifacts/vpd/; artifacts/sets/{val_0_64,val_0_8192,train_0_32768}.npy; E211's ctl inputs (E53's
# two states, E136/E143/E98/update_svd records); experiments/realistic_tasks/results/predictions_run2.json.
# Writes experiments/procedural_rl/results/{stage0,train_<task>,read_<task>}.json, results/read/<state>.json, and each
# state the moment it lands: results/large/states/<task>_<arm>_s<seed>.pt (bf16, ~57 MB, 9 per task).
#SBATCH --job-name=commute-procrl
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
export HF_HOME=$HOME/commute-data/cache/hf TMPDIR=$HOME/commute-data/cache/tmp; mkdir -p "$HF_HOME" "$TMPDIR"  # as E211/E222
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/procedural_rl/results/predictions.json || { echo "file the predictions first"; exit 1; }
[[ "${PRED_SHA:-SET_ME}" != SET_ME ]] || { echo "set PRED_SHA to the committed predictions.json's sha256"; exit 1; }
.venv/bin/python experiments/procedural_rl/run.py "$@"
