#!/bin/bash
# E153 (cell P1.11): E148's built control on XOR2 with a strong positive control (E148's plain seed-0 fine-tune as the
# original). Plan: reports/plans/2026-11-11_plan_built-control-strong.md. Filed rules:
# experiments/built_control_strong/results/predictions.json (the run refuses unless it is committed unchanged). One L40,
# debug QoS, $0. One job; exit 3 at the 105 min wall limit resumes from the saved files (resubmit the same line):
#   sbatch --comment=E153 cluster/built_control_strong.sbatch
# Times on an L40 from E148's logs (cluster/logs/commute-e148-57970.out, -58002.out): set-up 30 s, capacity read 120 s,
# 300 steps with its learned check ~110 s, H at five sites ~590 s, G at four sites 1,225 s; 3000 steps 810 s (E139's logs,
# as E148's sbatch). Expected ~52 min at the 300-step rung; ~60 min if seed 0 needs 3000 steps (G then at in2 only).
# Memory, from E139's logs as E148's sbatch states (E148's job 2 held two originals and a fine-tune, as here): host 6.1 GB
# peak; card about 35 GB at the H `final` site, under the L40's 48 GB.
# Inputs: E148's results/stage1.json and large/plain_s0.pt; E137's large/pieces.pt; E139's
# control.json (A1.setup checks the cue).
# Writes: results/stage0.json, stage1.json, strong_s{0,1}.json; results/large/strong_s{0,1}.pt (268 MB each, gitignored).
#SBATCH --job-name=commute-e153
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:55:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --comment=E153

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/built_control_strong/run.py
