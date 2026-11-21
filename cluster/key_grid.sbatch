#!/bin/bash
# E237 (cell P2.50): the five 96-key tasks at one common update size beside a random update of the same size; reads
# only, nothing trained. Plan reports/plans/2026-11-20_plan_p2-50-key-grid.md; rules
# experiments/key_grid/results/predictions.json (run.py refuses unless it is committed unchanged). Arguments go to run.py.
# Submit from the clone root, after ~/commute-agents/fleet/link-data . (results/large symlinks):
#   sbatch --comment=E237 cluster/key_grid.sbatch --stages 0 1 2                     (~55 min: stage 0 alone first)
#   sbatch --comment=E237 cluster/key_grid.sbatch --stages 3 --shard 0 2             (~60 min each; stages 2-4 exit 1 until stage 1 is done: submit them with --dependency=afterok:<that job>)
#   sbatch --comment=E237 cluster/key_grid.sbatch --stages 3 --shard 1 2
#   sbatch --comment=E237 cluster/key_grid.sbatch --stages 4 --shard 0 3 --tag v0    (VPD, ~60-90 min each on an L40)
#   (and --shard 1 3 --tag v1, --shard 2 3 --tag v2)
#   then, no GPU: ~/commute-agents/fleet/cpu-run python3 experiments/key_grid/score.py; ... plot.py
# A filed stop logs "STOP: ..." and exits 1; every stage resumes from what it has written.
# Peak host RSS of the readers imported: 6.1-6.2 GB (E221 run_pod0.log, E228 pod logs); peak GPU ~10 GB (E221, E226).
#SBATCH --job-name=commute-key-grid
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:50:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
for d in refit_96key key_grid topk_neurons transport_test; do
  test -L experiments/$d/results/large || { echo "experiments/$d/results/large is not linked: run ~/commute-agents/fleet/link-data ."; exit 1; }
done
.venv/bin/python experiments/key_grid/run.py "$@"
