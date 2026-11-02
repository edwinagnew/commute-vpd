#!/bin/bash
# E61 (qwen_recipe) Job S: the direct share against size on one cluster L40: experiments/qwen_recipe/size.py. Rules:
# experiments/qwen_recipe/results/predictions_size.json. Submit from the clone root, in this order; not submitted by the code
# writer. Exit 0 done; 1 a stop (results/size_stopped.json, or size2_stopped.json for the rehearsal; every stage refuses until
# it is removed) or a refusal; 3 the time budget was reached: resubmit the same command, it resumes from what is written.
# Stages: control; select (big and small states); state (one file per state; the scaled changes once search has written them);
# search (each random kind's scale to the big state); verdict; all (every one in order, state twice: the rehearsal's).
# Rehearsal, 2 rows, any time, Job R not needed: about 55 min in 3-4 debug jobs, resubmit on exit 3 (the resume is rehearsed):
#   sbatch --qos=debug --time=00:30:00 cluster/qwen_recipe_size.sbatch all --rows 2 --ladder gold_s0 --states 5 9
# Scored, after train_gold_1e-5.json has its G verdict and the rehearsal has passed:
#   sbatch --qos=debug --time=00:30:00 cluster/qwen_recipe_size.sbatch control   # 6 min
#   sbatch --qos=debug --time=00:30:00 cluster/qwen_recipe_size.sbatch select    # 8 min: 16 saved states at 20 s, the last 128 rows
#   sbatch cluster/qwen_recipe_size.sbatch state --budget-s 10000                # 125 min: 5 states, 3 of them with a split
#   sbatch --qos=debug --time=00:30:00 cluster/qwen_recipe_size.sbatch search    # 14 min: Gaussian, then alive
#   sbatch cluster/qwen_recipe_size.sbatch state --budget-s 10000                # 58 min: the 2 scaled changes
#   PYTHONPATH=experiments/qwen_reading:experiments/qwen_recipe ~/commute-agents/fleet/cpu-run --mem 8G \
#     python experiments/qwen_recipe/size.py --stage verdict                     # 2 min, CPU
# If normal is not free, `state` runs on debug too, without --budget-s: each job stops itself at 25 min, about 8 jobs in all.
# Time per state, from checks_k10_* logs (load 30 s, Sigma 171 s, Gaussian built 90 s once a job; 16 rows of cells 16 s, entering
# `applied` 26 s, split 355 s): cells 128 s + split 355 s (on file 15 s) + cells with split 154 s + true pass 16 chunks of
# 8 rows at 46 s, 12 min + both KLs 110 s = 26 min (20 min with the split on file: gold_s0, the Gaussian).
# Memory: GPU peak 42.43 GiB on 16 rows, the same at 128 (one row at a time); the rule stops at 44. Host: 80-85 GB in
# checks_k10_* (Sigma, Gamma parked, weight sets, X, 16 rows of masks). Here 8 rows x 2 mask sets holds the same 19 GB of masks
# and Gamma is back on the GPU then, so about 85 GB per state and 75 GB in search; 128G holds. `[true chunk]` logs a mask set's
# GB and the host peak per chunk; a Slurm kill at 128G loses one chunk, and the next job resumes.
#SBATCH --job-name=commute-qwen_recipe_size
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
stage=${1:?usage: sbatch cluster/qwen_recipe_size.sbatch control|select|state|search|verdict|all [--rows 2 --ladder NAME --states A B] [--budget-s S]}
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
export PYTHONPATH=$COMMUTE_REPO/experiments/qwen_reading:$PYTHONPATH   # E57's lab.py, score.py, boot_direct.py
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_recipe/size.py --stage "$stage" "${@:2}"   # the decomposition: COMMUTE_QWEN_DIR, env.sh
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
