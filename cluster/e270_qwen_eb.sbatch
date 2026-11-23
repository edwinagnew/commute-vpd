#!/bin/bash
# E270 (cell P3.14, qwen_eb) on one cluster L40: E_b of the six step-75 Qwen updates on p-9707d6ff. Arguments pass to
# experiments/qwen_eb/run.py. Submit from the clone root, after results/predictions.json is committed:
#   sbatch --comment=E270 cluster/e270_qwen_eb.sbatch --stages 0                    # the controls, alone
#   sbatch --comment=E270 --time=01:30:00 cluster/e270_qwen_eb.sbatch --stages 1    # the 12 items; a rerun resumes (time: set from stage 0's seconds per update x 12 + load + bases)
# Memory (estimated, measured by stage 0): GPU holds the fp32 model (8.1 GB), the decomposition's U, V (19.4 GB), the
# input second moments (5.6 GB) and one matrix's split in fp64 (~3.5 GB): ~37 of 44.3 GiB. Host: the split's bases
# with P, Q in fp32 (17 GB), W0's linear weights (5.6 GB), one state and its Gaussian change (11.3 GB): ~35 GB.
#SBATCH --job-name=commute-e270_qwen_eb
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub
nvidia-smi --query-gpu=name,memory.total --format=csv
trap 'rc=$?; echo "=== FINISHED (exit $rc) ==="' EXIT
.venv/bin/python experiments/qwen_eb/run.py --decomp "$HOME/commute-data/artifacts/qwen3-1.7b-p-9707d6ff-torch" "$@"
