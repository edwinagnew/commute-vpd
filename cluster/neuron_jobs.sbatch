#!/bin/bash
# E189 (cell P2.29): Goodfire-style neuron baselines on the nine built 67M pairs: do the neurons a fine-tune moves keep their
# Pile job (top-20 windows), and does its gain survive top-240 neurons per block? Plan: reports/plans/2026-11-16_plan_p2-29-neuron-jobs.md.
# Rule: experiments/neuron_jobs/results/predictions.json (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/compare-lead && ~/commute-agents/fleet/link-data compare-lead && sbatch --comment=E189 cluster/neuron_jobs.sbatch
# Stage 0 (controls on e46/ident32_s0, alone; three stops), stage 1 (68 saved updates), stage 2 (16 random updates, the calls);
# each prints "STAGE n DONE" after its results file. A filed stop prints "STOP: ..." and exits 1; stage 1 stops itself if the
# projected job time passes 9 min.
# Inputs on NFS (gitignored), read only: the updates and windows named in experiments/share_untrained/results/inventory.json
# and experiments/refit_96key/results/inventory.json, the cached Pile rows (artifacts/sets/val_0_8192.npy), the model in
# artifacts/vpd/. Writes experiments/neuron_jobs/results/stage{0,1,2}.json and results/large/pile_top20_w0.pt (~0.5 MB).
# Expected on an L40: about 5-7 min. E155 (job 58821) read 56 updates with two forwards each in ~1 min; this job adds four
# masked forwards per update on the same rows and a Pile pass of 2,000 x 512 tokens per update (stopped before the
# unembedding), ~2-4 s each, for 84 updates plus 6 controls. Host memory: one update (113 MB) at a time, E53's windows, the rows.
#SBATCH --job-name=commute-neuron-jobs
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/neuron_jobs/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/neuron_jobs/run.py
