#!/bin/bash
# E157 (cell P1.14): the sum-parity fine-tune read beside the last-number-odd control, between a planted ceiling and floor.
# Plan: reports/plans/2026-11-11_plan_sum-parity-beside-control.md; card: experiments/sum_parity_beside/card.txt. Filed
# rules: experiments/sum_parity_beside/results/predictions.json (the run refuses unless it is committed unchanged).
# One L40, debug QoS, $0. Submit from the clone root:
#   sbatch --comment=E157 cluster/sum_parity_beside.sbatch
# Exit 3 = the wall limit (6,300 s): resubmit with --comment="E157 resume: wall limit" (or fleet/resubmit --code 3); it
# resumes from results/{ceiling,repeat,floor,plain_s0,plain_s1}.json and results/large/plain_s{0,1}.pt.
# Exit 0 also at the card's stop (the planted ceiling's H at in2 under 0.5): ceiling.json then holds "card_stop".
# Time on an L40, per step, from E156's log (job 58855, cluster/logs/commute-e156-58855.out) and E154's (job 58754,
# cluster/logs/commute-e154-58754.out):
#   set-up ~25 s (E156: 22 s); class means ~10 s (one pass over 75,000 fit windows; E154's calibration ran 25 passes of
#   20,000 windows in 41 s, 534 s - 493 s);
#   calibration per planted edit ~25 bisection reads of 20,000 windows at ~3 s each, plus the whole read half ~15 s (E156:
#   F's gap 22 s -> 37 s): ~2 min;
#   one-site search (five sites) ~10 min per item (E156 control 37 s -> 630 s; E154 control_s0 677 s -> 1260 s);
#   repeat check ~3 min (gap ~15 s; one pair with its first fit, E156's in2+in3 ~125 s and in2+out3 111 s);
#   two sites (ten pairs) ~22 min per item (E156 b s0: 1224 s -> 2543 s, 1,319 s);
#   training 300 steps ~85 s with its checks (E154: 565 s -> 650 s), distances ~27 s (650 s -> 677 s); 3000 steps, if the
#   300-step rung is not learned, ~14 min (ten times 300 steps; not measured).
# Ceiling ~12 min + repeat 3 + ceiling two sites 22 + floor 12 + 22 + two seeds (~12 min each at 300 steps) + two seeds'
# two sites 44: ~2.3 h, ~2.8 h if both seeds need 3000 steps. So two jobs: the first exits 3 near 105 min, the second
# needs ~35-65 min (plus ~1 min of reload and refit).
# Memory: card peak ~35 GB at the `final` site (E154, E156; L40 46 GB), the same here. Host 6.2 GB peak in both E154 and
# E156 (their logs' last line), which also held a second copy of the 67M as here: --mem 16G as E156.
# Reads experiments/built_control_sum/results/amend1_{stage0,stage1}.json, results/large/amend1_control_s0.pt and
# experiments/h_diagnosis/results/ft_s0.json; writes only experiments/sum_parity_beside/results/.
#SBATCH --job-name=commute-e157
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:55:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --comment=E157

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/sum_parity_beside/run.py
