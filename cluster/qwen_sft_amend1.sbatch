#!/bin/bash
# E40 Amendment 1 on one cluster L40 (46 GB), debug QoS: stage L alone, or the --rehearse jobs. Plan:
# reports/plans/2026-10-28_plan_qwen-sft.md, Amendment 1 and its Gates; code experiments/qwen_sft/amend1.py.
# Submit from the clone root; every argument goes to amend1.py. The decomposition is read from commute-data. The out
# dir is on NFS, because the jobs land on different nodes and later ones read the earlier ones' files.
#   L alone (results/amend1/L.json; done 2026-10-28, keep it):
#     sbatch --mem=32G cluster/qwen_sft_amend1.sbatch --stages L --out /mnt/nw/home/e.agnew/commute-data/tmp/e40a1
#   the rehearsal (results/amend1_rehearse/; tiny sizes, F1 and L at full size). Clear both first, then five jobs,
#   each submitted when the one before has ended:
#     R=/mnt/nw/home/e.agnew/commute-data/tmp/e40a1_rehearse
#     rm -rf "$R" experiments/qwen_sft/results/amend1_rehearse experiments/qwen_sft/results/large/amend1_rehearse
#     sbatch cluster/qwen_sft_amend1.sbatch --rehearse --wandb-upload --stages B,G1,L,A0 --out $R      # ~25 min
#     sbatch --mem=32G cluster/qwen_sft_amend1.sbatch --rehearse --stages F1 --out $R                 # ~15 min
#     sbatch cluster/qwen_sft_amend1.sbatch --rehearse --wandb-upload --stages T0,D1,T1 --kill-after-save --out $R
#                                                       # ends with exit 3 after T1's first saved offsets (the kill)
#     sbatch cluster/qwen_sft_amend1.sbatch --rehearse --wandb-upload --stages T1,TR --out $R          # the resume
#     sbatch --mem=4G --time=00:05:00 cluster/qwen_sft_amend1.sbatch --rehearse --stages A0 --out $R
#                                                       # the stale-results guard: refuses at once (exit 1)
#   results/amend1_rehearse/rehearsal.json then lists the checks that missed beyond those expected at tiny size.
# --wandb-upload needs WANDB_API_KEY in the submitting shell (sbatch passes the environment); the rehearsal uploads
# only files under 50 MB. Expected peaks: GPU ~36-44 GB, host ~40 GB with U, V (bf16) and Γ parked; L 29.5 GB GPU and
# 10.9 GB host (measured); F1 ~37 GB GPU, ~20 GB host.
#SBATCH --job-name=commute-qwen_sft_amend1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_sft/amend1.py --decomp "$HOME/commute-data/artifacts/qwen3-1.7b-p-e9fcffd5-torch" \
  --volume none --usd-per-h 0 "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
