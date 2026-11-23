#!/bin/bash
# E_TBD (cell P2.66): the realistic convergence panel, stages 0-2 (stage 3, E_b, is eb.py on pods). Plan
# reports/plans/2026-11-22_plan_convergence-panel.md. Rules: experiments/convergence_panel/results/predictions.json
# (run.py refuses unless it is committed unchanged; the read stage also refuses until the raters are filed).
#   cd <clone> && ~/commute-agents/fleet/link-data <role or .>     (results/large must be the shared symlink)
#   sbatch --comment=E_TBD cluster/convergence_panel.sbatch 0                                   (alone, first)
#   for T in pile code lower british topic french yoruba squad conll:
#     sbatch --comment="E_TBD stage: train $T" --dependency=afterok:<stage-0 id> cluster/convergence_panel.sbatch train $T
#   ~/commute-agents/fleet/cpu-run python3 experiments/convergence_panel/run.py select          (after all nine)
#   for each kept T: sbatch --comment="E_TBD stage: read $T" cluster/convergence_panel.sbatch read $T
# Expected on an L40 (E211 log cluster/logs/commute-realistic-r2-61448.out: a 300-step LM fine-tune 10 s, lines 36;
# one full E211 read of a state, a superset of this panel's readings, 57-58 s, lines 12-13): stage 0 ~8 min (lowercase
# rows, three reads, E254's world); train ~5-10 min a task (3 seeds + up to 27 saves each, ~2 s a save; squad and
# conll add a twin and the HF downloads); read ~16 min a task (up to 12 states + 4 Gaussians at <= 58 s).
# Saves ~57 MB a state in bf16 under experiments/convergence_panel/results/large/states (<= 27 a run, ~40 GB at most
# for the panel; the selected ones, <= 108, are what stage 3 reads).
#SBATCH --job-name=commute-convergence-panel
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; HF caches in /ephemeral/$USER (env.sh)
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/convergence_panel/run.py "$@"
