#!/bin/bash
# E151 Amendment 1 (cell P2.17, the cell's last run): rerun after run 1 (job 58739) stopped on an empty common-word pool.
# Amendment: experiments/disagreeing_pairs/results/predictions_amend1.json (the common-word ladder, the realised-prior stop,
# pair B's premise); the filed rule results/predictions.json is unchanged. amend1.py refuses unless both are committed unchanged.
#   cd ~/commute-agents/compare-lead && ~/commute-agents/fleet/link-data compare-lead && sbatch --comment='E151 rerun: common-word ladder after empty pool' cluster/disagreeing_pairs_amend1.sbatch
# Stage 0: run 1's controls unchanged and first; the ladder's diagnostics logged ("[0 diagnostics]", "[0 rung n]",
# "[0 rung taken]", "[0 one-word draw]") before any table check; then the two tables, their realised priors and pair B's
# premise. Stages 1 and 2 as run 1's code. Each prints "STAGE n DONE" after its results file; a filed stop prints "STOP: ..."
# and exits 1; stages 1 and 2 stop themselves if the projected job time passes 25 min after four states.
# Inputs: as cluster/disagreeing_pairs.sbatch. Writes experiments/disagreeing_pairs/results/amend1_stage{0,1,2}.json (run 1's
# stage0.json is kept) and, in results/large/, the two new maps' windows and six bf16 updates dW_{common_cipher,one_word}_s{0,1,2}.pt.
# Expected on an L40: about 12 min (run 1's stage 0 plus one extra W0 pass over 32 held-out keys' windows); memory as run 1.
#SBATCH --job-name=commute-disagreeing-pairs-a1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/disagreeing_pairs/results/predictions_amend1.json || { echo "file the amendment first"; exit 1; }
test -d experiments/disagreeing_pairs/results/large/ || { echo "results/large is missing: run fleet/link-data"; exit 1; }
.venv/bin/python experiments/disagreeing_pairs/amend1.py
