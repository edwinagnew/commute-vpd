#!/bin/bash
# E103 (known_answer_pair) on one cluster L40, debug QoS, $0. Plan reports/plans/2026-11-06_plan_e103-known-answer-pair.md;
# rules experiments/known_answer_pair/results/predictions.json. Two jobs, in order, from the clone root:
#   sbatch cluster/known_answer_pair.sbatch run       # stages 0, T (C1 at the matched step), G
#   sbatch cluster/known_answer_pair.sbatch neurons   # after the first ends with exit 0: C2, then E102's readings
# Time on an L40. run: about 15 min. Stage 0 as E84's (83 s, experiments/qwen_tofu_forms/results/run.log:6); 335 steps at
# 0.9-1.4 s a step (E91's retrain, experiments/qwen_size_grid/results/run.log:61-64; E81's 1,250 steps in 1,793 s,
# experiments/qwen_tofu/results/run.log:25,87), so 5-8 min, plus about 30 candidate gain reads and two 5.64 GB saves;
# G 4 states at 24 s a load (qwen_size_grid/results/run.log:5-6) plus a 868-option read each. neurons: about 10 min,
# 7 loads and 10 reads of about 20 s on 128 rows (experiments/qwen_carrying/results/run.log:32). Limit 35 min each.
# --mem: E91's job, the same training and loads, peaked at 10.5 GB host (cluster/logs/commute-qwen_size_grid-52219.out:67);
# E102's second job at 10.5 GB (commute-qwen_carrying-53029.out:29). Exit 0 done; 1 a stop (results/stopped.json).
# E103 Amendment 1 (plan section "Amendment 1", rules results/predictions_amend1.json), same card and limits, after the
# above, in order:
#   sbatch cluster/known_answer_pair.sbatch amend1           # stages 0, R (C2, C1's bar), T (C1 at the matched step), G
#   sbatch cluster/known_answer_pair.sbatch amend1_neurons   # after amend1 ends with exit 0: C2n, the four matched rules
# amend1 about 10 min (E103's run took 406 s, job 53188; R and G add 5 loads); amend1_neurons about 3 min (E103's took 258 s
# for 7 loads, this has 2). Exit 1 writes results/stopped_amend1.json.
# E103 Amendment 2 (plan section "Amendment 2", rules results/predictions_amend2.json), same card and limits, in order:
#   sbatch cluster/known_answer_pair.sbatch amend2           # 0 (control 0, seed 142's first batch), R (C2, C1's bar), T, P
#   sbatch cluster/known_answer_pair.sbatch amend2_neurons   # after amend2 ends with exit 0: C2n, the set-identity rule
# amend2 about 12 min (Amendment 1's job took 493 s, job 53262; P adds one load, 2 x 868-option reads and 4 x 217 greedy
# answers); amend2_neurons about 2 min (Amendment 1's took 87 s for 2 loads and 3 reads; this the same). Limit 35 min each,
# about 3 and 15 times. Exit 1 writes results/stopped_amend2.json.
#SBATCH --job-name=commute-known_answer_pair
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:35:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
case "${1:-}" in
  run|neurons|amend1|amend1_neurons|amend2|amend2_neurons) ;;
  *) echo "give run, neurons, amend1, amend1_neurons, amend2 or amend2_neurons"; exit 1 ;;
esac
set +e
.venv/bin/python "experiments/known_answer_pair/$1.py"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
