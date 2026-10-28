#!/bin/bash
# E40 Amendment 1 on one cluster L40 (46 GB), debug QoS: stage L alone, or the --rehearse jobs. Plan:
# reports/plans/2026-10-28_plan_qwen-sft.md, Amendment 1 and its Gates; code experiments/qwen_sft/amend1.py.
# Submit from the clone root; every argument goes to amend1.py. The decomposition is read from commute-data. The out
# dir is on NFS, because the jobs land on different nodes and later ones read the earlier ones' files.
#   L alone (results/amend1/L.json; done 2026-10-28, keep it):
#     sbatch --mem=32G cluster/qwen_sft_amend1.sbatch --stages L --out /mnt/nw/home/e.agnew/commute-data/tmp/e40a1
#   the rehearsal (results/amend1_rehearse/; tiny sizes, F1 and L at full size). The first rehearsal (jobs 40602-40668)
#   is kept in results/amend1_rehearse_1/. Each rerun uses a new out dir and an empty results dir; four jobs, each
#   submitted when the one before has ended, then the guard check:
#     R=/mnt/nw/home/e.agnew/commute-data/tmp/e40a1_rehearse2
#     rm -rf experiments/qwen_sft/results/amend1_rehearse experiments/qwen_sft/results/large/amend1_rehearse
#     sbatch cluster/qwen_sft_amend1.sbatch --rehearse --wandb-upload --stages B,G1,T0,L,A0 --out $R   # ~17 min
#     sbatch --mem=32G cluster/qwen_sft_amend1.sbatch --rehearse --stages F1 --out $R                 # ~16 min
#     sbatch cluster/qwen_sft_amend1.sbatch --rehearse --wandb-upload --stages D1,T1 --kill-after-save --out $R
#                                                       # ~6 min; ends with exit 3 after T1's first saved offsets
#     sbatch cluster/qwen_sft_amend1.sbatch --rehearse --wandb-upload --stages T1 --out $R             # the resume, ~4 min
#     sbatch --mem=4G --time=00:05:00 cluster/qwen_sft_amend1.sbatch --rehearse --stages A0 --out $R
#                                                       # the stale-results guard: refuses at once (exit 1)
#   results/amend1_rehearse/rehearsal.json then lists every check and stop that missed (from checks.json, all jobs),
#   and those beyond the ones expected at tiny size.
# --wandb-upload needs WANDB_API_KEY in the submitting shell (sbatch passes the environment); it uploads the results
# JSONs and run.log at each job's end (offsets and δW stay on the volume), the rehearsal only files under 50 MB. Expected peaks: GPU
# ~36-45 GB, host ~40 GB with U, V (bf16) and Γ parked; L 29.4 GB GPU and 38 GB host, F1 36.7 GB GPU and 10.8 GB host
# (measured, jobs 40602, 40609).
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
