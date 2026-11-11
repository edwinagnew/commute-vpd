#!/bin/bash
# E156 (cell P1.13): why E154's no-teaching control reads H under 0.5. Plan: reports/plans/2026-11-11_plan_h-diagnosis.md;
# card: experiments/h_diagnosis/card.txt. Filed rules: experiments/h_diagnosis/results/predictions.json (the run refuses
# unless it is committed unchanged). One L40, debug QoS, $0. Submit from the clone root:
#   sbatch --comment=E156 cluster/h_diagnosis.sbatch
# Exit 3 = the wall limit (6,300 s): resubmit the same line, it resumes from results/control.json and results/ft_s{0,1}.json.
# Exit 0 also at the card's stop (H on the exact target F at in2 under 0.5): control.json then holds "card_stop".
# Time on an L40, from E154's log (job 58754, cluster/logs/commute-e154-58754.out): set-up ~30 s; the five-site H search
# ~550 s per item (ridge to `final` 39 s, then ~100 s per site). Here: control ~10 min; distances 2 x ~30 s; c 2 x ~4 min
# (one fit, four reads of ~45 s); a 2 x ~6 min; b 2 x ~23 min (10 pairs: ~60 s fit, up to ~40 s of solves, ~50 s read).
# About 75 min in all, ~105 min if the reads run 40% slower; a second job only if the first exits 3.
# Memory: card peak ~35 GB at E154's `final` site (L40: 46 GB), the same here (control at `final`, b's pairs ending at
# `final`); a's every-position fit to in3 ~22 GB. Host 6.1 GB peak in E154.
# Reads experiments/built_control_sum/results/amend1_{stage0,stage1,control_s0,control_s1}.json and
# results/large/amend1_control_s{0,1}.pt; writes only experiments/h_diagnosis/results/{control,ft_s0,ft_s1}.json.
#SBATCH --job-name=commute-e156
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:55:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --comment=E156

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/h_diagnosis/run.py
