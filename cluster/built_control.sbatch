#!/bin/bash
# E148 (cell P1.9): the built control on XOR2, plain 67M against a 67M taught XOR2. Plan: reports/plans/2026-11-10_plan_built-control-xor2.md.
# Filed rules: experiments/built_control/results/predictions.json (the run refuses unless it is committed unchanged). One L40, debug QoS, $0.
# Submit from the clone root, job 2 only after job 1 printed "STAGE 1 DONE" and exited 0:
#   sbatch --comment='E148 job1' --time=01:00:00 cluster/built_control.sbatch job1   (stage 0 controls, stage 1 checks and the learned rung: ~15 min, ~45 min worst case)
#   sbatch --comment='E148 job2' cluster/built_control.sbatch job2                      (H 2 seeds per arm, G seed 0: ~95 min; exit 3 at the 105 min wall limit, read saved)
# Times on an L40 from E139's logs (jobs 57673, 57745, 57831): 300 steps 82 s, 3000 steps 810 s, H at five sites ~590 s, G at four sites 1,370 s.
# Inputs: E137's results/stage0.json, large/pieces.pt; E139's results/control.json (A1.setup checks the cue).
# Writes: results/stage0.json, stage1.json, {taught,plain}_s{0,1}.json; results/large/taught_original.pt, plain_s0.pt (268 MB each, gitignored).
# Memory, from E139's logs: host 6.1 GB peak; card about 35 GB at the H `final` site.
#SBATCH --job-name=commute-e148
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:55:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --comment=E148

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/built_control/run.py "$@"
