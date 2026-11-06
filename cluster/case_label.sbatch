#!/bin/bash
# E112, the case-label pair, one stage per job: 0 (W0: pool, windows, possession probe and control, P0, W0's removals),
# 1 (the four fine-tunes, delta-W in bf16 to results/large/), 2 (do-nothing first, then P1-P3 and verdicts; pruning and
# reverse fine-tuning beside). Plan: reports/plans/2026-11-06_plan_e112-case-label.md. Rules:
# experiments/case_label/results/predictions.json (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && ~/commute-agents/fleet/link-data <role> && sbatch cluster/case_label.sbatch 0
# Then `... sbatch cluster/case_label.sbatch 1` once stage 0 exits 0, and `... 2` once stage 1 exits 0.
# Exit 0 done; 1 a stop (stage 0: P0 fails, verdict 'unreadable'; stage 2: the do-nothing direction over 0.10 on a state);
# 3 (stage 1 only) too little time left for the next fine-tune: resubmit stage 1, finished states are kept.
# Inputs: Pile train rows 0-32767 and 32768-65535 (cached in artifacts/sets), the target model in artifacts/vpd/.
# Writes experiments/case_label/results/stage<N>.json (after every step) and stage<N>.out (this log, appended);
# results/large/task.pt (windows, ~5 MB), probes.pt (probes and Pile windows, ~3 MB), dW_{real,twin}_s{0,1}.pt (bf16, ~57 MB each).
# Expected time on a debug L40 (estimates, not yet measured for this code): stage 0 about 5 min (model load ~1, rows and
# tokenizer ~1, 9,600 + 1,600 residual windows < 1, ridge fits seconds); stage 1 up to ~20 min (E58's arm-A steps 0.21-0.29 s,
# so 1,000 steps 3.5-5 min per state, less when a state reaches the bar early); stage 2 about 10 min. Memory: E99's 67M jobs
# peaked at 6.1 GB host; 16G as theirs.
#SBATCH --job-name=commute-e112
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
.venv/bin/python experiments/case_label/run.py --stage "$1" 2>&1 | tee -a "experiments/case_label/results/stage$1.out" || rc=$?
echo "=== FINISHED E112 stage $1 (exit $rc) — results in $COMMUTE_REPO/experiments/case_label/results/ ==="
exit $rc
