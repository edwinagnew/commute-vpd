#!/bin/bash
# E154 (cell P1.12, Amendment 1 of E150): the built control on the parity of a sum, rerun with the plain-model capacity bar
# at 0.9. Plan: reports/plans/2026-11-11_plan_built-control-sum.md ("Amendment 1, E154").
# Filed rules: experiments/built_control_sum/results/predictions_amend1.json (the run refuses unless it is committed unchanged).
# One L40, debug QoS, $0. Submit from the clone root (exit 3: resubmit the same job, it resumes):
#   sbatch --comment=E154 cluster/built_control_sum_amend1.sbatch job1               (stage 0, stage 1, control arm, taught original: ~60 min, ~100 min worst)
# then, once job1 has exited 0, jobs 2 and 3 in parallel (each reads only job1's files and writes its own):
#   sbatch --comment='E154 stage: 2' cluster/built_control_sum_amend1.sbatch job2   (plain arm: ~47 min, ~60 min worst; refuses if the control reads below)
#   sbatch --comment='E154 stage: 3' cluster/built_control_sum_amend1.sbatch job3   (taught arm: ~47 min; exits 0 at once if the taught arm was dropped)
# Times on an L40 from E148's logs (jobs 57970, 58002): see cluster/built_control_sum.sbatch.
# Writes results/amend1_{stage0,stage1,taught_original}.json, amend1_{control,plain,taught}_s{0,1}.json; results/large/
# amend1_taught_original.pt, amend1_{arm}_s{seed}.pt (268 MB each, gitignored). E150's files are not touched.
# Memory, from E148's logs: host 6.1 GB peak; card about 35 GB at the H `final` site (L40: 46 GB).
#SBATCH --job-name=commute-e154
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:55:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --comment=E154

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/built_control_sum/amend1.py "$@"
