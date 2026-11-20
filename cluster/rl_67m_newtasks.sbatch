#!/bin/bash
# E224 (cell P2.41 runs 6-7): E220's four arms and E222's readings on three task#2191 tasks (1 plural pronoun, 2 day/month
# sequence, anchor addition). Plan: reports/plans/2026-11-19_plan_rl-67m.md (Stage 2c/3c (E224)). Settings, expected
# values and rules: experiments/rl_67m/results/predictions_E224.json (both scripts refuse unless it is committed
# unchanged). Arguments go to run_newtasks.py (training), or with a leading `read` to read_newtasks.py (readings).
# Submit from the clone root, one job per line; job ids replace <...>:
#   run 6 (training)
#   sbatch --comment='E224 rehearse: newtasks ctl' --time=00:20:00 cluster/rl_67m_newtasks.sbatch ctl              # first and alone
#   sbatch --comment=E224 --time=00:20:00 --dependency=afterok:<ctl> cluster/rl_67m_newtasks.sbatch null 1e-05     # and 3e-05, 0.0001, 0.0003
#   sbatch --comment=E224 --time=00:15:00 --dependency=afterok:<ctl> cluster/rl_67m_newtasks.sbatch arm 1 rl 0     # and arm 2 rl 0, arm anchor rl 0
#   sbatch --comment=E224 --time=00:15:00 --dependency=afterok:<ctl> cluster/rl_67m_newtasks.sbatch arm 1 rl 1 2   # and arm 2 rl 1 2
#   sbatch --comment=E224 --time=00:15:00 --dependency=afterok:<rl 0 of 1 and 2> cluster/rl_67m_newtasks.sbatch arm 1 own   # and 2 own, 1 demo, 2 demo, anchor demo
#   sbatch --comment=E224 --time=00:15:00 --dependency=afterok:<rl 0s and all four nulls> cluster/rl_67m_newtasks.sbatch arm 1 shuffled   # and 2
#   run 7 (readings), after every arm job
#   sbatch --comment=E224 --time=00:15:00 cluster/rl_67m_newtasks.sbatch read ctl                                  # first and alone
#   sbatch --comment=E224 --time=00:20:00 --dependency=afterok:<read ctl> cluster/rl_67m_newtasks.sbatch read 1   # and read 2, read anchor
# Times on an L40, each a measured item x its count: model load 0.5-2 min (E220 log 61914: 28 s; E222 ~2 min) plus the
# items (plural mining over <= 32,768 Pile rows, measured by ctl, estimated <= 1 min): ~3 min per job. ctl ~8 min (3 +
# reads and 10 timed steps ~1 + E211's ctl 4, E222 log 61971). An RL step samples one token, so ~0.1-0.5 s against E220's
# ~4 s (ctl measures it); a held-out read (512 items, their controls, 64 KL rows) ~2 s. null ~4-8 min (two tasks x 30-300
# steps; 13 min if both run 600); arm with one seed ~4 min, three seeds ~6 min; anchor rl 0 <= 8 min (600 steps).
# Readings: per state ~23 s (E222: 18 s at 200 rows of 65; 256 rows here); read ctl ~5 min (3 + three readers and zero
# reads at ~0.6 min, E222 ctl); read 1 and 2 ~9 min (3 + 0.6 + 13 reads x 23 s); read anchor ~5 min. Host memory:
# E211's ctl 6.1 GB peak (E222); the rest under 10 GB.
# Inputs: the model in artifacts/vpd/; artifacts/sets/{val_0_64,val_0_8192,train_0_32768}.npy; ctl also E211's inputs
# (E53's two states, E136/E143/E98/update_svd records). Writes experiments/rl_67m/results/stage2c_<job>.json,
# results/large/stage2c/<task>_<arm>_s<seed>.pt (bf16, ~57 MB each, 28 states, ~1.6 GB), stage3c_<job>.json and
# results/stage3c/<task>_<arm>_s<seed>.json.
#SBATCH --job-name=commute-rl67m-nt
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
export HF_HOME=$HOME/commute-data/cache/hf TMPDIR=$HOME/commute-data/cache/tmp; mkdir -p "$HF_HOME" "$TMPDIR"  # as E222: l40-worker's /ephemeral was full (2026-11-18)
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/rl_67m/results/predictions_E224.json || { echo "file the predictions first"; exit 1; }
if [[ "$1" == read ]]; then
  .venv/bin/python experiments/rl_67m/read_newtasks.py "${@:2}"
else
  .venv/bin/python experiments/rl_67m/run_newtasks.py "$@"
fi
