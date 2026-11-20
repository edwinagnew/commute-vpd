#!/bin/bash
# E228 (cell P2.44): the re-gating share on random weight changes of the 67M scaled to four saved fine-tunes' KL on their
# read rows. Plan reports/plans/2026-11-20_plan_p2-44-random-kl-matched.md. Settings, expected values and rules:
# experiments/random_kl_share/results/predictions.json (run.py refuses unless it is committed unchanged). ctl first and
# alone (exit 1 on a miss), then one draws job per fine-tune after it:
#   cd <clone> && ~/commute-agents/fleet/link-data .
#   sbatch --comment=E228 cluster/random_kl_share.sbatch ctl
#   for s in c3/copy_s0 c3/cipher_s0 e62/lookup_s0 e53/lowercase_s0:
#     sbatch --comment=E228 --dependency=afterok:<ctl id> cluster/random_kl_share.sbatch draws $s
#   then, no GPU: ~/commute-agents/fleet/cpu-run python3 experiments/random_kl_share/run.py score
# Writes experiments/random_kl_share/results/{ctl.json,draws_<state>.json,score.json}; prints "STAGE ctl DONE" or "STAGE
# draws <state> DONE"; a filed stop prints "STOP: ..." and exits 1; a rerun of draws skips the draws written.
# Inputs: the model in artifacts/vpd/, the four saved updates and their read windows (E136's inventory), cached Pile rows.
# Times on an L40, each a measured item x its count: model load ~1 min; the share ~2 s per 64 rows (E161, job 59106), so
# ~2 s (75 rows), ~6 s (200), ~16 s (500) per read; a KL evaluation is two plain forwards (no masks), well under the
# share's four; a draw is at most 52 KL evaluations (perturb.match: 12 grid points, 40 halvings) plus one share read and
# one KL. ctl ~3 min (zero change, four states x three share reads and two KLs); draws (ten each) ~2 min (c3 states),
# ~3 min (e62), ~6 min (e53). Peak GPU under 10 GB, host under 10 GB (E211 ctl 6.1 GB). 20-min limit.
#SBATCH --job-name=commute-random-kl-share
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
test -f experiments/random_kl_share/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/random_kl_share/run.py "$@"
