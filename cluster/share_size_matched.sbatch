#!/bin/bash
# E204 (cell P2.34): the re-gating share of three fine-tunes imitating E137's Always on push scaled to Table's gain, beside
# E202's Table readings. Plan: reports/plans/2026-11-17_plan_share-size-matched.md. Rules:
# experiments/share_size_matched/results/predictions.json (the run refuses unless it is committed unchanged). One L40, debug QoS, $0.
# Submit from the clone root:
#   sbatch --comment='E204 share size matched' cluster/share_size_matched.sbatch
# Stops (exit 1): stage 0 identity, stage 1 positive control, stage 2 alpha scan (no repeat of E202's alpha 1 windows,
# saturation, poor match, kind check), stage 3 step-10 check or a seed under the reach bar. A rerun resumes stage 3 from saved states.
# Inputs: experiments/reuse_variable_test/results/{stage0.json,large/pieces.pt}; experiments/regating_table/results/large/
#   {windows_ident32.pt,ident32_s0.pt}; experiments/method_table/results/final.json; experiments/share_built_kind/results/run2/
#   {stage2.json,stage3.json}; the Pile rows and the 67M as E200/E202.
# Writes: experiments/share_size_matched/results/{stage0..3,rules}.json, always_on_big_s{0,1,2}_train.json (small);
#   results/large/{always_on_big_s{0,1,2}.pt (~268 MB each), rows_big.pt}.
# Time on an L40: about 15 min (rows ~1, controls ~1, alpha scan ~2-5, three fine-tunes ~1.5 each plus distances and reads ~1 each).
#SBATCH --job-name=commute-e204
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
# (E200, which also fine-tuned three states, peaked at 14.8 GB host RSS under 16G)
#SBATCH --time=00:45:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/share_size_matched/run.py
