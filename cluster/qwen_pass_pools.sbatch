#!/bin/bash
# E120 (qwen_pass_pools) stage 3 on one cluster L40, $0: experiments/qwen_pass_pools/train.py, one arm per job. Needs
# experiments/qwen_pass_pools/results/{pools_amend2.json,predictions.json} and the results/large link (fleet/link-data). Submit
# from the clone root; not submitted by the code writer. Arms may run at the same time (each has its own files).
# Amendment 2 (2026-11-06): the arms read results/pools_amend2.json (MATH regraded), passed as --pools.
# Normal QoS (the Qwen line may use it when the queue is empty at submission, .claude/CLAUDE.md → Rules):
#   sbatch cluster/qwen_pass_pools.sbatch known 0 --pools experiments/qwen_pass_pools/results/pools_amend2.json
#   sbatch cluster/qwen_pass_pools.sbatch known 1 --pools experiments/qwen_pass_pools/results/pools_amend2.json
#   sbatch cluster/qwen_pass_pools.sbatch new 0 --pools experiments/qwen_pass_pools/results/pools_amend2.json
#   sbatch cluster/qwen_pass_pools.sbatch new 1 --pools experiments/qwen_pass_pools/results/pools_amend2.json
# Debug QoS (2 h a job, `scontrol show assoc_mgr flags=qos`); the budget leaves 10 min for the last unit and the exit:
#   sbatch --qos=debug --time=02:00:00 cluster/qwen_pass_pools.sbatch known 0 --budget-min 110 \
#       --pools experiments/qwen_pass_pools/results/pools_amend2.json
# Exit 0 done; 1 a stop (results/stopped_<arm>.json; that arm refuses until it is removed) or a refusal; 3 paused, the
# work so far saved: resubmit the same command (fleet/resubmit --code 3). Extra arguments go to train.py (--t-chunk N).
# --mem: E61's gold arm peaked at 11.0 GB on the host (train_gold_1e-5.json → peaks); this run holds one more fp32 δW
# (5.6 GB, the step before a bracket) beside the one being saved, so about 17 GB; 24G.
#SBATCH --job-name=commute-qwen_pass_pools
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
pool=${1:?usage: sbatch cluster/qwen_pass_pools.sbatch known|new 0|1 [train.py arguments]}
seed=${2:?usage: sbatch cluster/qwen_pass_pools.sbatch known|new 0|1 [train.py arguments]}
shift 2
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$pool $seed $*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_pass_pools/train.py --pool "$pool" --seed "$seed" "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
