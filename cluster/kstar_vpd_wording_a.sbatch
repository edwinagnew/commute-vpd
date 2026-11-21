#!/bin/bash
# E??? (cell pending, `fleet/cell approve 42`): R under wording A on E226's six built edits and E187's nine pairs at
# matched size, $0 on the cluster's debug QoS. Fifteen jobs, each resumable (an item whose JSON exists is skipped):
#   sbatch --comment='E??? known0' cluster/kstar_vpd_wording_a.sbatch known0 --items new_thin
#   sbatch --comment='E??? known1' cluster/kstar_vpd_wording_a.sbatch known1 --items new_few
#   sbatch --comment='E??? known2' cluster/kstar_vpd_wording_a.sbatch known2 --items x2 x10 moved rebuilt
#   for i in $(seq 0 11); do sbatch --comment="E??? pairs$i" cluster/kstar_vpd_wording_a.sbatch pairs$i --pairs --shard $i 12; done
# Each job reads the do-nothing state first (stops on a miss). PRED_SHA is the sha256 of the committed predictions.json
# (`git show HEAD:experiments/kstar_vpd_wording_a/results/predictions.json | sha256sum`), set here once it is filed.
# After the last job: .venv/bin/python experiments/kstar_vpd_wording_a/run.py --summary (on the dev node via cpu-run).
#SBATCH --job-name=commute-wording-a
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
tag=${1:?tag, e.g. known0 or pairs3}
shift
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
( PRED_SHA=FILL_IN_WHEN_FILED .venv/bin/python experiments/kstar_vpd_wording_a/run.py --tag "$tag" "$@" ) 2>&1 \
    | tee -a "experiments/kstar_vpd_wording_a/results/run_$tag.log" || rc=${PIPESTATUS[0]}
echo "=== FINISHED E??? $tag (exit $rc) — results in $COMMUTE_REPO/experiments/kstar_vpd_wording_a/results/ ==="
exit $rc
