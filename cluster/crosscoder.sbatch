#!/bin/bash
# P2.31 (E198): BatchTopK crosscoders with latent scaling, base 67M against each saved fine-tune of the nine built pairs,
# as built and size-matched. Rules: experiments/crosscoder/results/predictions.json (run.py refuses unless committed
# unchanged and numbered). Plan: reports/plans/2026-11-17_plan_p2-31-crosscoder.md.
#   cd ~/commute-agents/compare-lead && ~/commute-agents/fleet/link-data compare-lead && sbatch --comment=E198 cluster/crosscoder.sbatch
#       stages 0 and 1 (self and planted controls, then e54_matched seed 0, the random update, a crosscoder reseed):
#       6 crosscoders; "STAGE 0 DONE", "STAGE 1 DONE"; a filed stop logs "STOP: ..." and exits 1.
#   sbatch --comment='E198 stage 2' cluster/crosscoder.sbatch 2 0/5    # ... 2 4/5: stage 2 in five shards, after stage 1 passed
#       106 crosscoders over the shards, one file each in experiments/crosscoder/results/sides/; resumes from them.
# Inputs on NFS (gitignored): artifacts/vpd/, artifacts/sets/{train_0_32768,val_0_256}.npy, the states'
# results/large/ of regating_table, keydependent_elicit, open_pair, induction_cue, performed_anchor, refit_96key, and the
# task windows they name (experiments/share_untrained/results/inventory.json, refit_96key's inventory).
# Time: not measured; estimated 3-5 min per crosscoder on an L40 (plan, "Cost"); the first side logs its steps and the
# stage-2 shard stops itself if its projection passes 110 min.
#SBATCH --job-name=commute-crosscoder
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -L experiments/refit_96key/results/large || { echo "results/large is not linked: run ~/commute-agents/fleet/link-data compare-lead"; exit 1; }
.venv/bin/python experiments/crosscoder/run.py "$@"
