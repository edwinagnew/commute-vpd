#!/bin/bash
# E101, carrying-neuron jobs on the 67M, stage 0 (W0 only): top Pile windows of E98's carrying neurons and of block-matched
# random neurons, the already-engaged share, and stage 1's hashed batch files.
# Plan: reports/plans/2026-11-05_plan_e101-carrying-jobs.md. Rules: experiments/carrying_jobs/results/predictions.json
# (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/carrying_jobs.sbatch            # stage 0: experiments/carrying_jobs/run.py
#   cd ~/commute-agents/<role> && sbatch cluster/carrying_jobs.sbatch maximise   # stage 1's scoring job (maximise.py: C2, then the agents' texts; ~3 min)
#   cd ~/commute-agents/<role> && sbatch cluster/carrying_jobs.sbatch controls_amend1   # Amendment 1: controls, writer batch c00, task windows (< 5 min)
# Stage 0 exits 1 on a missed control (C0_pile, C0_w0, C0_null, C0_files), each logged before the next step.
# Inputs on NFS (gitignored): the task windows E98 read (experiments/{regating_table,keydependent_elicit,performed_anchor,
# open_pair,open_pair_instruments,induction_cue}/results/large/), the target model in artifacts/vpd/, Pile val rows
# 0-8191 in artifacts/sets/val_0_8192.npy; E98's experiments/neuron_r/results/carrying.json (committed).
# Writes experiments/carrying_jobs/results/stage0.json (small) and results/large/{pile_top.pt,engaged.pt,key.json,
# batches/b*.json}; commits nothing.
# Expected: 3-6 minutes. E98's job 52786 (cluster/neuron_r.sbatch) took about 4 minutes for 56 states at 7 forwards per
# batch plus two 300-step fine-tunes (reports/analyses/2026-11-05_analysis_e98-neuron-r.md:289); this job is W0 forwards
# only: 2,160 Pile rows, 13 tasks' read windows once, and one sort of a 12288 x 65536 sample (about 16 GB peak on the GPU).
# One L40, l40-worker's only card type (cluster/README.md:3); other settings as cluster/neuron_r.sbatch.
#SBATCH --job-name=commute-carrying-jobs
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
test -f experiments/carrying_jobs/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python "experiments/carrying_jobs/${1:-run}.py"
