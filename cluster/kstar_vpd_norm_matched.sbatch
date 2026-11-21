#!/bin/bash
# E240 (P1.44): N, N_w, N_in and R on the six built edits rescaled to new_thin's update size, $0 on the cluster.
# Three jobs at once, one per shard of predictions.json "shards" (0: new_thin, 1: new_few, 2: x2 x10 moved rebuilt):
#   sbatch --comment='E240 job0: N at matched size' cluster/kstar_vpd_norm_matched.sbatch 0
#   sbatch --comment='E240 job1: N at matched size' cluster/kstar_vpd_norm_matched.sbatch 1
#   sbatch --comment='E240 job2: N at matched size' cluster/kstar_vpd_norm_matched.sbatch 2
# Each job reads the do-nothing state first (stops on a miss); job 0 reads new_thin at c = 1, the reproduction, first.
# A rerun skips items whose JSON exists. The last job to end writes results/summary.json.
#SBATCH --job-name=commute-e240-matched
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
shard=${1:?shard index 0, 1 or 2}
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
( PRED_SHA=1c08427862de0b1b0211b3014edb7919654bd30f725d6e7e83e0ee639c2224ce .venv/bin/python experiments/kstar_vpd_norm_matched/run.py --shard "$shard" --tag "job$shard" ) 2>&1 \
    | tee -a "experiments/kstar_vpd_norm_matched/results/run_job$shard.log" || rc=${PIPESTATUS[0]}
echo "=== FINISHED E240 job$shard (exit $rc) — results in $COMMUTE_REPO/experiments/kstar_vpd_norm_matched/results/ ==="
exit $rc
