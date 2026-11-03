#!/bin/bash
# E69 (qwen_rl): the $0 recipe probe on one cluster L40, before the paid run. Submit from the clone root; arguments go to
# experiments/qwen_rl/probe.py:
#   sbatch cluster/qwen_rl_probe.sbatch --arm R --lr 1e-6 --warmup 10
#   sbatch cluster/qwen_rl_probe.sbatch --f-step --lr 3e-7
# Rules in experiments/qwen_rl/results/predictions_probe.json; results in results/probe_<name>.json, every number also in this
# job's log. Needs no key: it uploads nothing. Exit 0 done; 1 a refusal or an error.
# --time: 20 steps at about 150 s, 5 reads and the pool. --mem as cluster/qwen_rl_sanity.sbatch; the weights before a read step
# are held on the host, 5.6 GB. All the cluster's cards are L40s.
#SBATCH --job-name=commute-qwen_rl_probe
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_rl/probe.py "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
