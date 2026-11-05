#!/bin/bash
# E99 stage 4, part (d): the class map and its shuffled twin rebuilt (E58 Amendment 7's windows, 30 steps of E54's recipe, seeds
# 0 and 1) and read under the value form of C = the embedding class at blocks 1 and 2, control first: C's teeth and the do-nothing
# state alone, then the block-1 hand-write control, then R-adm (b)'s block-2 reader, then the states.
# Plan: reports/plans/2026-11-05_plan_e99-refine-verify-definition.md. Rules: experiments/definition_verify/results/
# predictions_stage4.json (stage4.py refuses unless it is committed unchanged). Parts (a)-(c) are stage4_cpu.py, no GPU.
#   cd ~/commute-agents/<role> && ~/commute-agents/fleet/link-data <role> && sbatch cluster/definition_verify_stage4.sbatch
# Exit 1 at a stop: windows sha256 off E58's, a step-30 held-out P(target) off E58's by more than 0.02, no matched band, or a
# block-1 hand write losing more than 0.2 of its gain. No teeth is a reading and does not stop.
# Inputs: open_pair/results/large/amend7/windows_class_{map,shuffled}_trained.pt; open_pair/results/amend7.json; new_vs_reuse
# results (predictions.json settings, stage0.json labels); Pile train rows 32768-65535 (cached).
# Writes experiments/definition_verify/results/stage4_class.json, appends stage4.out; results/large/dW_class_{map,shuffled}_s{0,1}.pt
# (bf16, ~57 MB each). Deletes nothing, commits nothing.
# Expected time on a debug L40, from measured runs: start-up, Pile rows and E90's capability for 32 keys about 1 min (E95 job
# 52507, cluster/logs/commute-e95-52507.out lines 1 and 7: 15:19:31 to 15:19:56), its cache and matched draws 24 s (line 8,
# 15:20:20), W0's control 15 s (line 10, 15:20:35); here 128 keys, so x4: about 3 min. Fine-tunes: E58's four reached step 30 in 6.4-8.8 s
# (open_pair/results/large/amend7.out, the "checkpoint 30" line before lines 146, 226, 390, 465), so under 1 min with the copies. Writes and reads:
# E90 Amendment 2 read 16 states and 16 writes at two blocks under 20 draws in under 2 min (cluster/logs/commute-e90-52444.out,
# 14:45:25 to 14:47:26); here 4 states, 6 writes, 2 sites of 286 windows: about 1 min. Total about 6 min; the limit is 30 min.
# Memory: E90 and E95 peaked at 6.1 GB host (their logs' last line), as did stages 0-3 (job 52807); the fit residuals for 128 keys
# (up to 256,000 windows x 4 blocks x 768, fp32, 3.1 GB) sit on the GPU, the window tokens on the host add under 0.1 GB. 16G
# leaves over twice the measured peak.
#SBATCH --job-name=commute-e99-stage4
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
.venv/bin/python experiments/definition_verify/stage4.py 2>&1 | tee -a experiments/definition_verify/results/stage4.out || rc=$?
echo "=== FINISHED E99 stage 4 (d) (exit $rc) — results in $COMMUTE_REPO/experiments/definition_verify/results/ ==="
exit $rc
