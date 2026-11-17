#!/bin/bash
# Qwen instrument check, step 1 (cell P3.3, E152): planted and random changes read with E132's pipeline on the new
# decomposition p-9707d6ff (experiments/qwen_instrument_check/step1.py; plan
# reports/plans/2026-11-11_plan_qwen-instrument-check-step1.md). One L40, debug QoS, $0. Through a pod-runner:
#   cd ~/commute-agents/qwen-lead
#   sbatch --comment='E152 stage: step 1' cluster/qwen_instrument_check_step1.sbatch
# Time, from E132's job 56867 on the same card and pipeline (experiments/qwen_regating_new/results/run.log): loads 95 s
# (lines 2-4); W0 cells on 128 rows 118 s (line 5); zero split 44 s, E40 cells and do-nothing on 16 rows ~45 s
# (lines 7-12); Σ 176 s (line 16); one split 361-372 s (lines 47, 81, 114, 146, 178, 210); cells on 128 rows 108-133 s
# and the split's cells 146-160 s (lines 19-211). Here: controls ~6 min, Σ 3 min; per kind one split ~6.5 min, three
# searches of up to 8 batched KL passes (~20 s each, estimated: 16 weight copies of 7 GB plus 256 forwards of 8 rows)
# ~4-8 min, three reads of ~4.6 min; ~25-29 min per kind, ~85-97 min in all. Exit 3 = cut short before the 2 h limit
# with what is written; resubmit the same line to resume (a resume reloads ~3 min and redoes the cut kind's split).
# Memory, E132's peaks: GPU 42.43 GiB in the split's cells (line 48; Lab stops above 44 after row 0), host 91.1 GB
# (line 47). The search parks Γ and holds two 1.2 GB logit batches; the planted kinds share one zero tensor per shape.
# Log lines to wait on: "STAGE controls DONE", "STAGE <kind>_<size> DONE" (nine), "STAGE summary DONE".
#SBATCH --job-name=commute-qwen_instrument_check_step1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub   # the model snapshot is here already
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_instrument_check/step1.py --decomp "$COMMUTE_QWEN_NEW_DIR"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
