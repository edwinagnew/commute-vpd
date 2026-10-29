#!/bin/bash
# E44 (qwen_maps) $0 rehearsal on one cluster L40 (46 GB), debug QoS. Plan: reports/plans/2026-10-28_plan_qwen-q0q1.md,
# stage R; code experiments/qwen_maps/run.py. Submit from the clone root; every argument goes to run.py. The decomposition
# and E40's small files (g statistics, sources, step-0 answers) are read from commute-data; --out is on NFS, so a later
# job (the resume) sees the earlier one's files. Not submitted by the code writer; the pod-runner submits, in order:
#   1. stage R at full size (results/R.json; Gram against SVD on layer 14 first, then fp32 ranks of all 392 U and V):
#        sbatch --mem=48G cluster/qwen_maps.sbatch --stages R --out /mnt/nw/home/e.agnew/commute-data/tmp/e44_R
#      R.json "holds" false -> the pod run adds --svd-bases (plan section 4). The ranks may outrun 30 min: the check
#      is written first, and the log shows the ranks' pace.
#   2. every other stage at REHEARSAL sizes (2 rows of 64 tokens per set, batch 1, random δW on layer 0, bf16
#      components, 2-step adversary, 4 GSM8K problems; results/rehearsal/). Stops are obeyed, as on the pod, so the
#      stop and resume paths run first (a rehearsal from code before 2026-10-28's review fixes: move its
#      results/rehearsal/ aside and use a new --out):
#        R=/mnt/nw/home/e.agnew/commute-data/tmp/e44_rehearse2
#        sbatch cluster/qwen_maps.sbatch --rehearse --stages C,S,D,K,M,G,A,H --out $R
#          -> expect "STOP control_W0" (random δW, 2 rows), exit 1, results/rehearsal/stopped.json
#        sbatch --time=00:05:00 cluster/qwen_maps.sbatch --rehearse --stages C,S,D,K,M,G,A,H --out $R
#          -> refuses at once: stopped.json exists
#        sbatch cluster/qwen_maps.sbatch --rehearse --waive control_W0,teeth --stages C,S,D,K,M,G,A,H --out $R
#          -> "WAIVED" lines for control_W0 (and teeth if it misses), then every stage to the end
#   3. the resume: the last command again; every stage logs "done earlier: skipped" (C and K re-read their gates,
#      logged WAIVED), within a minute of load.
# Host memory: the alive columns (14.6 GB fp32 in R, 7.3 GB bf16 in the rehearsal) plus Σ and δW for one layer.
#SBATCH --job-name=commute-qwen_maps
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_maps/run.py --decomp "$COMMUTE_QWEN_DIR" \
  --e40-large "$HOME/commute-data/results-large/qwen_sft/nowarmup" --usd-per-h 0 "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
