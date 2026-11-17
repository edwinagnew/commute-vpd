#!/bin/bash
# E159 (cell P1.15): the difficulty ladder, the rewiring share read as graded beside linear and MLP probes of the plain 67M.
# Plan: reports/plans/2026-11-11_plan_difficulty-ladder.md; card: experiments/difficulty_ladder/card.txt. Filed rules:
# experiments/difficulty_ladder/results/predictions.json (the run refuses unless it is committed unchanged).
# One L40, debug QoS, $0. Submit from the clone root:
#   sbatch --comment=E159 cluster/difficulty_ladder.sbatch
# Exit 3 = the wall limit (6,300 s): resubmit with --comment="E159 resume: wall limit" (or fleet/resubmit --code 3); it
# resumes from results/{repeat,probes,target3,ceiling,floor,sum3_s0,sum3_s1}.json and results/large/sum3_s{0,1}.pt.
# Exit 1 = a stop (the JSON of the stage holds "stopped") or a refusal.
# Time on an L40, per stage, from E157's log (job 58950, cluster/logs/commute-e157-58950.out):
#   set-up ~35 s (E157: 31 s, plus the second window set);
#   repeat ~2.5 min (class means 5 s, 31 s -> 36 s; gap ~15 s; one pair with its first fit at in1, E157's in1+in3 122 s);
#   probes ~10 min, not measured: 80 probes (5 targets x 4 sites x linear/MLP x real/shuffled) of at most 20,000 Adam steps
#   of a 768 -> 100 -> 1 net at ~0.5 ms a step; shuffled MLPs run all 200 epochs, real ones stopped at 24-103 epochs on a
#   CPU toy; worst case ~13 min;
#   ceiling and floor on the triples ~11 min each (calibration 84 s, 36 s -> 120 s; five sites 578 s, 120 s -> 698 s);
#   each sum-of-three seed ~11.5 min at 300 steps (training, check, distances 110 s, 4088 s -> 4198 s; five sites 573 s),
#   ~14 min more if it needs the 3000-step rung;
#   two sites (ten pairs) ~21.5 min per item (E157 ceiling 860 s -> 2143 s; floor 2812 s -> 4088 s).
# Total ~2.4 h (~2.9 h if both seeds need 3000 steps): job 1 exits 3 near 105 min during the planted floor's two-site
# reading; job 2 needs ~45-70 min.
# Memory: card peak ~35 GB at the `final` site (E154, E156, E157; L40 46 GB); the probes' states add ~1.2 GB per layout and
# are freed before the searches. Host 6.2 GB peak in E157 (its log's last line), same model copies here: --mem 16G as E157.
# Reads experiments/built_control_sum/results/amend1_{stage0,stage1}.json and experiments/sum_parity_beside/results/
# ceiling.json; writes only experiments/difficulty_ladder/results/.
#SBATCH --job-name=commute-e159
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:55:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --comment=E159

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/difficulty_ladder/run.py
