#!/bin/bash
# step_split (E19): the step split along the emoticon fine-tune (plan: reports/2026-10-21_plan_step-split.md). The
# arguments go to run.py:
#   cd ~/commute-agents/<role>
#   sbatch --job-name=commute-step_split-smoke --time=00:20:00 cluster/step_split.sbatch all --smoke
#   sbatch cluster/step_split.sbatch all            # needs results/predictions.json with "status": "filed"
#   sbatch cluster/step_split.sbatch all --resume   # after a lost job: re-runs the fine-tune, scores the missing states
# Before that: `cluster/sync.sh up` from the laptop (code and the DRAFT/filed predictions file), round 1's alive.pt and
# windows.pt under experiments/transport_test/results/large on the cluster. Results land in the NFS repo copy
# (experiments/step_split/results/, dumped after every state); `cluster/sync.sh down` brings them home.
# Expected on one L40 (scaled from E16's H100 log, and from E7/E9's L40 Pile timings of ~115 s per state): load and the
# W0 control ~7 min (3 attacker seeds), fine-tune with both step splits ~3 min, 9 Pile states ~18 min, CPU tables ~2 min:
# ~30 min; the limit below gives ~1.8x. Smoke ~6 min.
#SBATCH --job-name=commute-step_split
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/step_split/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/step_split/run.py "$@"
