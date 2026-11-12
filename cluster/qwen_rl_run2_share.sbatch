#!/bin/bash
# P3.4 run 2, the outside share of every read state against a Gaussian change of the same norms, and |δW| beside the
# decomposition's leftover norm, by E152's step-0 code (read2.py --stage share), CPU only, debug QoS, $0. Submit from the
# clone root once every state is read (results/run2_reads/reads.json):
#   sbatch --comment=E167 cluster/qwen_rl_run2_share.sbatch
# Time: E152 step 0 measured ~30 min of eigh for the 84 MLP matrices plus ~10 min for the rest on 8 CPUs (its sbatch),
# with 15 updates; here up to 18 states plus the control, ~100 GB read one matrix at a time from NFS. The job exits 3
# when the time left is under 1.5 x the longest matrix: resubmit the same command until the log says STAGE share DONE
# (expect 2-3 jobs). Host: ~2 GB per matrix (E152), 16G leaves room for the larger update set.
#SBATCH --job-name=commute-qwen_rl_run2_share
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=""
set +e
.venv/bin/python experiments/qwen_rl_readable/read2.py --stage share --wall-min 28 "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
