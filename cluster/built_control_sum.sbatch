#!/bin/bash
# E150 (cell P1.10): the built control on the parity of a sum of two numbers. Plan: reports/plans/2026-11-11_plan_built-control-sum.md.
# Filed rules: experiments/built_control_sum/results/predictions.json (the run refuses unless it is committed unchanged). One L40, debug QoS, $0.
# Submit from the clone root, each job after the one before exited 0 (exit 3: resubmit the same job, it resumes):
#   sbatch --comment=E150 cluster/built_control_sum.sbatch job1                 (stage 0, stage 1, control arm, taught original: ~60 min, ~100 min worst)
#   sbatch --comment='E150 stage: 2' cluster/built_control_sum.sbatch job2     (plain arm: ~47 min, ~60 min worst; refuses if the control reads below)
#   sbatch --comment='E150 stage: 3' cluster/built_control_sum.sbatch job3     (taught arm: ~47 min; exits 0 at once if the taught arm was dropped)
# Times on an L40 from E148's logs (jobs 57970, 58002): capacity read ~115 s, 300 steps 82 s, 3000 steps 810 s (E139),
# H at five sites ~550 s, G at four sites ~1,340 s, row set-up 20 s.
# Inputs: E137's results/stage0.json, large/pieces.pt; E139's results/control.json (A1.setup); E148's results/stage1.json
# and large/plain_s0.pt (stage 0's read).
# Writes: results/stage0.json, stage1.json, taught_original.json, {control,plain,taught}_s{0,1}.json; results/large/
# taught_original.pt, {arm}_s{seed}.pt (268 MB each, gitignored).
# Memory, from E148's logs: host 6.1 GB peak; card about 35 GB at the H `final` site (L40: 46 GB).
#SBATCH --job-name=commute-e150
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:55:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --comment=E150

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/built_control_sum/run.py "$@"
