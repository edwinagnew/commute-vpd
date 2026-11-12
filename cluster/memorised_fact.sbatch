#!/bin/bash
# E168 (cell P1.16): a memorised fact against its reuse twin, the re-weighting share G read on both.
# Plan: reports/plans/2026-11-12_plan_memorised-fact.md; card: experiments/memorised_fact/card.txt. Filed rules:
# experiments/memorised_fact/results/predictions.json (the run refuses unless it is committed unchanged).
# One L40 per job, debug QoS, $0. The argument is the item (E168_ONLY); none runs every item in order. Submit from the
# clone root, the control first and alone:
#   sbatch --comment=E168 cluster/memorised_fact.sbatch planted_odd
# then, once its log shows "STAGE planted_odd PASSED" (the first G at least 0.5; every other item refuses before that),
# the five in parallel:
#   sbatch --comment=E168 cluster/memorised_fact.sbatch planted_fact      (and floor, twin_s1, fact_s0, fact_s1)
# fact_s1 waits up to 40 min for fact_s0's rung (exit 3 if it is not written). After the last:
#   ~/commute-agents/fleet/cpu-run --mem 6G env E168_ONLY=readings python experiments/memorised_fact/run.py
# Exit 3 = the wall limit (6,300 s): resubmit the same line with --comment="E168 resume: wall limit"; it resumes from
# results/<item>.json and results/large/fact_s{0,1}.pt. Exit 1 = a stop (the item's JSON holds "stopped") or a refusal.
# Time on an L40 (E154 and E157 logs): set-up ~35 s; one G search (12 starts, 104-117 s each) ~22 min; one H search
# (five sites) ~10 min; calibration ~1.5 min; training at 300 steps with its checks and distances ~2 min (~14 min more at
# 3000 steps). Per item: planted_odd ~23 min (passes after ~4-8 min, at in2), planted_fact ~35, floor ~23, twin_s1 ~23,
# fact_s0 ~35 (~50 at 3000 steps), fact_s1 the same plus its wait.
# Memory: card peak ~35 GB at the `final` H site (E154, E157; L40 46 GB); host ~6 GB: --mem 16G as E157 and E159.
# Reads experiments/built_control_sum/results/amend1_{stage0,stage1,control_s0,control_s1}.json and large/
# amend1_control_s1.pt, experiments/sum_parity_beside/results/{ceiling,floor}.json; writes only
# experiments/memorised_fact/results/.
#SBATCH --job-name=commute-e168
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:55:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --comment=E168

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export E168_ONLY="${1:-${E168_ONLY:-}}"
.venv/bin/python experiments/memorised_fact/run.py
