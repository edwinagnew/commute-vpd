#!/bin/bash
# E190 (cell P2.30): E189 (neuron_jobs, stopped on its top-K check, job 60130) rerun with gain kept at k 240, 768 and 1536.
# Plan: reports/plans/2026-11-16_plan_p2-30-neuron-jobs-k.md. Code imports experiments/neuron_jobs (frozen as it ran).
# Rule: experiments/neuron_jobs_k/results/predictions.json (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/compare-lead && sbatch --comment=E190 cluster/neuron_jobs_k.sbatch
# Stage 0 (controls on e46/ident32_s0, alone: zero update and retention stop on a miss; the top-K check per k and W0 under
# the mask per k are recorded), stage 1 (68 saved updates), stage 2 (16 random updates, the calls); each prints
# "STAGE n DONE" after its results file. A filed stop prints "STOP: ..." and exits 1; stage 1 stops itself if the projected
# job time passes 20 min.
# Inputs on NFS (gitignored), read only: the updates and windows named in experiments/share_untrained/results/inventory.json
# and experiments/refit_96key/results/inventory.json, the cached Pile rows (artifacts/sets/val_0_8192.npy), the model in
# artifacts/vpd/. Writes only experiments/neuron_jobs_k/results/stage{0,1,2}.json.
# Expected on an L40: about 10-12 min. E189's stage 0 (job 60130) took 53 s from start, model load and W0 Pile pass
# included, for 6 reads (~5 s each with the Pile pass); this job makes ~90 reads with 6 masked forwards each instead of 2.
# Host memory: one update (113 MB) at a time, E53's windows, the rows (E189 peaked at 6.1 GB).
#SBATCH --job-name=commute-neuron-jobs-k
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/neuron_jobs_k/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/neuron_jobs_k/run.py
