#!/bin/bash
# E160 (cell P3.5, run 1): experiments/qwen_pair_library/run.py, one stage per job, one L40, $0. Plan
# reports/plans/2026-11-11_plan_p35-qwen-pair-library.md. Through a pod-runner, from the clone root:
#   sbatch --comment='E160 stage: capacity' cluster/qwen_pair_library.sbatch capacity
#   sbatch --comment='E160 stage: floors' --mem=128G cluster/qwen_pair_library.sbatch floors
#   Amendment 1 (results/predictions_amend1.json), one job per side, in parallel:
#   sbatch --comment='E160 rerun: rehearse at lr 2e-6, new stop rule, side copy' --qos=normal --time=03:00:00 cluster/qwen_pair_library.sbatch rehearse --side copy
#   sbatch --comment='E160 rerun: rehearse at lr 2e-6, new stop rule, side cipher' --qos=normal --time=03:00:00 cluster/qwen_pair_library.sbatch rehearse --side cipher
# (rehearse is planned for an H100 pod instead: `python experiments/qwen_pair_library/run.py rehearse --side copy` (or cipher), with
# results/floors.json synced and results/large a real folder.)
# Memory and time:
#   capacity  fp32 model only (~7 GB GPU); 6 sides x 256 short greedy answers plus the priors; estimated 10-15 min. 24G host.
#   floors    the decomposition card (Decomp: fp32 target, components, Γ; >= 46 GB card): E132's W0 cells took 118 s on
#             128 rows after 95 s of loads (cluster/qwen_instrument_check_step1.sbatch header); 9 row sets (template 64
#             rows) ~ 18 min + loads ~ 25 min. Host 128G as that job (E132 peaked at 91 GB with the split; floors reads
#             no split).
#   rehearse  E120's pool_sft peaked 32 GiB GPU, 12.4 GB host at seq 512 (facts file §1); the copy rows are ~25 tokens.
#             Up to 100 steps with the KL read every step on 128 rows of 256, ending right after the 15x read; up to 4
#             fp32 saves of 5.64 GB (two brackets) per side under results/large; estimated 20-45 min a side. 24G host as
#             qwen_pass_pools (one fp32 δW on the host at a time).
# Log lines to wait on: "STAGE <stage> DONE". Exit 1: a stop (results/stopped.json; rehearse: results/stopped_<side>.json)
# or a refusal.
#SBATCH --job-name=commute-qwen_pair_library
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
stage=${1:?usage: sbatch cluster/qwen_pair_library.sbatch capacity|floors|rehearse [--side copy|cipher]}
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub   # the model snapshot is here already
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
shift
.venv/bin/python experiments/qwen_pair_library/run.py "$stage" --decomp "$COMMUTE_QWEN_NEW_DIR" "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
