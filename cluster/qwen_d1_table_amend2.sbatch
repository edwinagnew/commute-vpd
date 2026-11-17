#!/bin/bash
# E93 Amendment 2 (qwen_d1_table): the table and the adjoint split on the NEW decomposition p-9707d6ff, cluster L40, $0:
# experiments/qwen_d1_table/amend2.py. Plan: reports/plans/2026-11-09_plan_e93-new-decomposition.md; rules:
# experiments/qwen_d1_table/results/predictions_amend2.json. Needs p-9707d6ff in `GSTATS_OF` / `ALIVE_TOTAL_OF`
# (src/commute/qwen_d3.py, after E131 Amendment 1 stage 1) or stage 0 stops at the KeyError. Submit from the clone root,
# one job at a time, each after the last has ended (not submitted by the code writer):
#   sbatch cluster/qwen_d1_table_amend2.sbatch --stage control                                    # ~35 min, alone
#   sbatch cluster/qwen_d1_table_amend2.sbatch --states a2_s42,a2_s142 --map ls                    # ~50 min
#   sbatch cluster/qwen_d1_table_amend2.sbatch --states a1_s0,a1_s1,u1,u2 --map ls                 # ~1.0 h
#   sbatch cluster/qwen_d1_table_amend2.sbatch --states R_s0,R_s70,N_s0,N_s70,F_s0 --map ls        # ~1.0 h
#   sbatch cluster/qwen_d1_table_amend2.sbatch --extras R10,u3,tofu_E81                            # ~30 min
#   sbatch cluster/qwen_d1_table_amend2.sbatch --states a1_s0,a1_s1,a2_s42,a2_s142 --map adjoint   # ~50 min
#   sbatch cluster/qwen_d1_table_amend2.sbatch --states u1,u2,R_s0,R_s70,N_s0,N_s70,F_s0 --map adjoint  # ~1.2 h
# Exit 0 done; 1 a stop (results/amend2/stopped.json; every run refuses while it exists) or a refusal; 3 time left under
# the longest state so far: resubmit the same command (fleet/resubmit --code 3, one job at a time).
# Times are the old decomposition's on this card, less the map-free reads (carried): experiments/qwen_d1_table/results/
# run.log (E93: "done in", state seconds 944-1820; its map-free reads about 340-410 s per state, run.log:6-23, 188-205),
# results/amend1/run.log (Amendment 1, "done in", 544-780 s). New: E131's timing is the only one on new (26 min per 128
# rows of adversary, unused here); the split time depends on the new alive count, not yet read (w0_refs_stage1.json).
# Total ~5.7 h of card, 7 jobs, each under 2 h (--time guard: the run writes as it goes and exits 3 before the limit).
# Scoring (CPU, after any job): ~/commute-agents/fleet/cpu-run python experiments/qwen_d1_table/amend2_score.py
# Card: E93 peaked at 42.7 GiB with the same passes; stage 0's 512-token W0 read about 1.6 GiB above that. Stops over 44
# GiB (256-token rows). Host: E93's 95.6 GB, Amendment 1's 93 GB with Σ; the run stops over 105 GB.
#SBATCH --job-name=commute-qwen_d1_table_amend2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
set +e
.venv/bin/python experiments/qwen_d1_table/amend2.py --decomp "$COMMUTE_QWEN_NEW_DIR" "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
