#!/bin/bash
# E107 Amendment 1 (label_vs_style): E107's wrong-answer matched states through E94's VPD reads, on one cluster L40,
# debug QoS, $0: experiments/label_vs_style/amend1_vpd.py. Plan: reports/plans/2026-11-06_plan_e107-amend1-vpd-sets.md;
# rules: experiments/label_vs_style/results/predictions_amend1.json. One job, from the clone root:
#   sbatch cluster/label_vs_style_amend1.sbatch --states wrong_s42_matched,wrong_s142_matched
# then, on CPU: ~/commute-agents/fleet/cpu-run python experiments/label_vs_style/amend1_sets.py
# Time: E94 Amendment 1's known-facts matched states took 1627 s and 1501 s each (experiments/qwen_d3_rows/results/run.log
# lines 278, 302), of which the own-row passes (known-facts and TOFU rows) were ~630 s (lines 262-272); this run skips
# them, so ~16-17 min a state (split ~400 s, maths base ~135 s, maths new pass ~240 s, load and sha256 ~3 min), start-up
# ~3 min: ~37 min for both. --time stays 2 h because run.py's time guard (Config.limit_s) assumes the debug limit.
# Card and host as E94's (peaks 42.5 GiB, 80 GB): --mem=110G, the run stops over 44 GiB and 105 GB.
# Exit 0 done; 1 a stop (experiments/qwen_d3_rows/results/stopped.json; every E94 run refuses while it exists) or a
# refusal; 3 the time left was under the longest state so far: resubmit the same command, written states are skipped.
#SBATCH --job-name=commute-label_vs_style_amend1
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
.venv/bin/python experiments/label_vs_style/amend1_vpd.py --decomp "$COMMUTE_QWEN_DIR" "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
