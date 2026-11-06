#!/bin/bash
# E109 (larger_size) on one cluster L40, debug QoS, $0. Plan reports/plans/2026-11-06_plan_e109-neurons-larger-size.md;
# rules experiments/larger_size/results/predictions.json. Two jobs, in order, from the clone root:
#   sbatch cluster/larger_size.sbatch reads    job A: stage 0 (the control, first and alone), stage 1 (existing states)
#   sbatch cluster/larger_size.sbatch wrong    job B: the wrong arm to step 335 at seeds 42 and 142; refuses until A is done
# Time on an L40, from the logs. E107 job 53575 (experiments/label_vs_style/results/run.log): C2n 10.7 s (lines 1-3), a
# 217-sentence read 2.1-2.5 s (lines 4-5), a fp32 state's real-probe read 4.6 s and TOFU read 20.4 s (lines 52-53), a
# fp32 save 17 s (line 54). E103 Amendment 2 job 53316 (experiments/known_answer_pair/results/amend2.log): a 335-step
# arm with its candidate reads and the end save 340 s (line 9 at 62.6 s to line 67 at 402.9 s); greedy answers 12.6 s for
# two suffixes (lines 68-69). A span probe (about 42 rows of 256) is taken as 7 s, a state load as 8 s.
# Job A: start-up and model about 1 min; stage 0 about 1.5 min (C2n, W0, four states x (load + 2 sentence reads));
# stage 1 13 states, 26 probe reads: about 6.5 min. About 9 min; limit 30 min.
# Job B: per seed 340 s training and save, then sentences, greedy (one suffix) and five probe reads, about 75 s: 7 min
# a seed, about 16 min with start-up; limit 30 min.
# --mem as E107's (host peak 11.0 GB). Exit 0 done; 1 a stop (results/stopped.json). A rerun skips finished states.
#SBATCH --job-name=commute-larger_size
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
set +e
.venv/bin/python experiments/larger_size/run.py "$1"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
