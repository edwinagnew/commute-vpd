#!/bin/bash
# E54, a key -> value map W0 already half-knows beside a norm-matched cipher; steering S against the route split's r.
# Plan: reports/plans/2026-10-29_plan_e54-keydependent-elicit.md. Two jobs, the second only after the first passed:
#   cd ~/commute-agents/<role>
#   sbatch cluster/keydependent_elicit.sbatch 01   # stage 0 (controls, alone; exit 1 on a miss), stage 1 (W0's values,
#                                                  # key draw, rates; exit 1 on a miss); commits stage0.json and
#                                                  # stage1.json (only those paths) so stage 1 is filed before any fine-tune
#   sbatch cluster/keydependent_elicit.sbatch 2    # refuses unless stage1.json is committed unchanged: the four fine-tunes,
#                                                  # floors, S and r, the filed rules
# Inputs on NFS (gitignored): E46's ident32_s0.pt, rand32_s0.pt, windows_{ident32,rand32}.pt in
# experiments/regating_table/results/large/; experiments/transport_test/results/large/alive.pt; Pile rows
# artifacts/sets/train_0_32768.npy; the target model and decomposition in artifacts/vpd/. Writes small JSON to
# experiments/keydependent_elicit/results/ and windows (job 1) and four to six weight changes (job 2, ~113 MB each) to its
# results/large/ (cipher_own_s* is not saved when it is the same state as cipher_s*). predictions.json must be committed
# before job 2. Expected: job 1 ~3-6 min, job 2 ~6-12 min (four 300-step fine-tunes, 32 checkpoints held in RAM).
#SBATCH --job-name=commute-keydependent-elicit
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
R=experiments/keydependent_elicit/results
test -f $R/predictions.json || { echo "file the predictions first ($R/predictions.json)"; exit 1; }
case "${1:-}" in
  01)
    .venv/bin/python experiments/keydependent_elicit/run.py --stage 01
    git add $R/stage0.json $R/stage1.json
    git commit -m "E54 stages 0-1: controls and key selection (job $SLURM_JOB_ID)" -- $R/stage0.json $R/stage1.json \
      || echo "COMMIT FAILED: commit $R/stage0.json and $R/stage1.json by hand before job 2"
    ;;
  2) .venv/bin/python experiments/keydependent_elicit/run.py --stage 2 ;;
  *) echo "usage: sbatch cluster/keydependent_elicit.sbatch 01|2"; exit 2 ;;
esac
