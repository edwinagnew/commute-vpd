#!/bin/bash
# R2 (cell P1.4): fine-tunes of all weights of the 67M imitating the four E137 pairs, 3 seeds. Plan: reports/plans/2026-11-09_plan_r2-finetunes-imitating-pairs.md.
# Rules and cut-offs: experiments/reuse_variable_finetune/results/predictions.json (the run refuses unless it is committed unchanged). The lead replaces the
# placeholder experiment number below (and `EXPERIMENT` in run.py) when it claims the run. One L40, debug QoS, $0. Submit from the clone root:
#   sbatch cluster/reuse_variable_finetune.sbatch control                 (stage 0, alone and first: about 6 min; exit 1 if it fails)
#   sbatch cluster/reuse_variable_finetune.sbatch always_on:0 cue:0 table:0   (three items per job fit the 1:55 limit; several jobs run side by side)
# Items are pair:seed, with pair in always_on, cue, xor, table and seed in 0, 1, 2. Exit 3 means the wall limit (105 min) was reached with results saved: resubmit
# the same line, it resumes after the last finished search. Inputs: R1's results (experiments/reuse_variable_test/results/stage0.json, stage2.json, large/pieces.pt).
# Writes: results/control.json, results/<pair>_s<seed>.json, results/large/<pair>_s<seed>.pt (268 MB each, gitignored).
# Time on an L40, from R1's logs (jobs 57043, 57100): item about 35 min; memory about 35 GB of the card at the `final` site, as E137.
#SBATCH --job-name=commute-r2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:55:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --comment=E???

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/reuse_variable_finetune/run.py "$@"
