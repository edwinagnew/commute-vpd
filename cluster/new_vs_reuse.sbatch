#!/bin/bash
# E90, the new-against-reuse definition, frozen; a miss is a gap. Plan: reports/plans/2026-11-05_plan_e90-new-vs-reuse-definition.md,
# "Scope as restated" (two debug jobs). Stage 1's input-dependent tasks are not designed yet (stage1.py `input_dependent`). Rules: experiments/new_vs_reuse/results/predictions.json
# (the scripts refuse unless it is committed unchanged). Argument: stage1 or stage2, in that order; stage 2 refuses without a
# complete stage 1, and either refuses while results/stop.json exists (a sha256 mismatch or the readout's stops).
#   cd ~/commute-agents/<role> && sbatch cluster/new_vs_reuse.sbatch stage1
# Inputs on NFS (gitignored): results/large of regating_table, open_pair, induction_cue, task_ladder, reading_a_coverage; Pile rows
# artifacts/sets/train_32768_32768.npy (cached, E59, E64); the target model in artifacts/vpd/.
# Writes experiments/new_vs_reuse/results/stage{1,2}.json (small), results/large/{readout,bases,X_*}.pt (hand writes 9 MB each).
# Expected: stage 1 ~10 min. E64's cache of ~57K windows and its readout took 1-2 min and the job about 10 (cluster/readout64.sbatch
# header); here ~60K scanned windows, re-cut windows, 2 readouts, 3 controls on 2,000 windows, 7 states, 3 hand writes. Stage 2 ~10 min
# (plan's appendix: ~15 states + E62's six; here 21 states and 15 hand writes), exit 3 at a 22-min deadline, then resubmit. --mem as cluster/readout64.sbatch:27. 30-min wall.
#SBATCH --job-name=commute-e90
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
.venv/bin/python "experiments/new_vs_reuse/$1.py"
