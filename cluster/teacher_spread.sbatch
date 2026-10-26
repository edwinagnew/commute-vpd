#!/bin/bash
# teacher_spread (E37): teacher training of a fine-tune with no trigger (E36's lowercase fine-tune). Plan:
# reports/plans/2026-10-26_plan_teacher-spread.md.
#   cd ~/commute-agents/<role>
#   sbatch cluster/teacher_spread.sbatch     # needs experiments/teacher_spread/results/predictions.json (filed first)
# Inputs on NFS: experiments/transport_test/results/large/{alive.pt,windows.pt} (round 1, gitignored). Pile train rows
# 0..4095 and val rows 0..255 are read from artifacts/sets/ or streamed (~2 min). Results land in the clone
# (results/teacher_spread.json after every stage, results/large/run.log, three ~60 MB bf16 copies in results/large/);
# a rerun resumes from the JSON on the same card. C1 checks this card's known W0 floor (L40: 0.2088).
# Expected on one L40, scaled from E34's H100 brief times by 1.5-2: load and C1 ~6 min, rows ~2 min, teacher ~1 min,
# do-nothing and adjoint ~4 min, three 1000-step copies with a harness read each ~15-20 min, budget adversary ~3 min:
# ~33-38 min; the limit
# gives ~1.8x. Peak GPU memory estimated under 25 GB (logged per stage).
#SBATCH --job-name=commute-teacher_spread
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/teacher_spread/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/teacher_spread/run.py "$@"
