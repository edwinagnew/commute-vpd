#!/bin/bash
# E61 (qwen_recipe) Job S: the direct share against size on one cluster L40: experiments/qwen_recipe/size.py. Rules:
# experiments/qwen_recipe/results/predictions_size.json. Submit from the clone root, in this order; not submitted by the code
# writer. Exit 0 done; 1 a stop (results/size_stopped.json, or size2_stopped.json for the rehearsal; every stage refuses until
# it is removed) or a refusal; 3 the time budget was reached: resubmit the same command, it resumes from what is written.
# Stage more: exploration after the verdict, 3 more gold states, 80 min on normal, --budget-s 10000 (results/predictions_size_more.json).
# Stages: control; select (big and small states); state (one file per state; the scaled changes once search has written them);
# search (each random kind's scale to the big state); verdict; all (every one in order, state twice: the rehearsal's).
# Rehearsal, 2 rows at 1 row a chunk, any time, Job R not needed: about 70 min in 4 debug jobs, resubmit on exit 3. In the
# first job's true pass, `scancel` after the first `true_from_row` log line and resubmit, so a resume inside the pass runs:
#   sbatch --qos=debug --time=00:30:00 cluster/qwen_recipe_size.sbatch all --rows 2 --ladder gold_s0 --states 5 9
# Afterwards remove its 7 splits, 39 GB: results/large/checks/size2_*_X.safetensors.
# Scored, after train_gold_1e-5.json has its G verdict and the rehearsal has passed; about 3 h 50 of jobs, in this order:
#   sbatch --qos=debug --time=00:30:00 cluster/qwen_recipe_size.sbatch control   # 6 min
#   sbatch --qos=debug --time=00:30:00 cluster/qwen_recipe_size.sbatch select    # 8 min: 16 saved states at 20 s, the last 128 rows
#   sbatch cluster/qwen_recipe_size.sbatch state --budget-s 10000                # 137 min: 5 states, each with its own split
#   jq '.states[.small.name].step' experiments/qwen_recipe/results/size_select.json   # T, the small state's step
#   sbatch --qos=debug --time=00:30:00 cluster/qwen_recipe_train.sbatch G --run gold_1e-5 --step T   # 6 min: its drop and length
#   sbatch --qos=debug --time=00:30:00 cluster/qwen_recipe_size.sbatch search    # 14 min: Gaussian, then alive
#   sbatch cluster/qwen_recipe_size.sbatch state --budget-s 10000                # 58 min: the 2 scaled changes
#   PYTHONPATH=experiments/qwen_reading:experiments/qwen_recipe ~/commute-agents/fleet/cpu-run --mem 8G \
#     python experiments/qwen_recipe/size.py --stage verdict                     # 2 min, CPU; refuses without the G read
# If normal is not free, `state` runs on debug too, without --budget-s: each job stops itself at 25 min, about 9 jobs in all.
# Time per state, from checks_k10_* logs (load 30 s, Sigma 171 s, Gaussian built 90 s once a job; 16 rows of cells 16 s, entering
# `applied` 26 s, split 355 s): cells 128 s + split 355 s + cells with split 154 s + true pass 16 chunks of 8 rows at 46 s,
# 12 min + both KLs 110 s = 26 min. Every state runs its own split, saved and read in fp32 (review M1): 5.6 GB a state, 39 GB
# for 7. A rerun of `search` gives a new b, so a new split file, and `state` stops on the old state_*_big.json: remove it.
# Memory: GPU peak 42.43 GiB on 16 rows, the same at 128 (one row at a time); the rule stops at 44. Host: 80-85 GB in
# checks_k10_* (Sigma, Gamma parked, weight sets, X, 16 rows of masks). Here 8 rows x 2 mask sets holds the same 19 GB of masks
# and Gamma is back on the GPU then, so about 85 GB per state and 75 GB in search. A named stop at 118 GB after a chunk's masks
# (rules.host_stop_gb, size_stopped.json), before Slurm's kill at 128G; `control`'s 8-row chunk is the first real reading, at
# minute 6: the rehearsal's 2 rows hold 4.8 GB of masks and cannot check it. `[true chunk]` logs the host peak per chunk.
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
stage=${1:?usage: sbatch cluster/qwen_recipe_size.sbatch control|select|state|more|search|verdict|all [--rows 2 --ladder NAME --states A B] [--budget-s S]}
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
