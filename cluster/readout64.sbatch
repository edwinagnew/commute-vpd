#!/bin/bash
# E64: a linear readout per block of the 67M model over the pair's 64 table values, trained on Pile positions; W0's own
# possessed set as its reference; the removal test on the twelve saved E54/E56 fine-tunes. Plan:
# reports/plans/2026-11-02_plan_baseline-next3-draft.md (option 1, "Amendment before filing (2026-11-02)"). Rules:
# experiments/readout64/results/predictions.json (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/readout64.sbatch
# Stage 0 runs first and the job exits 1 on any miss, before any removal on a fine-tuned state is scored: inputs reproduce
# E54/E56; W0's possessed set (elicit |M| >= 8, cipher |M| <= 2); positions per value (a value under 200 is unfit; > 8
# unfit in a task stops); readouts trained and saved, early checks read off a batch sd; the construction check; the
# random-direction share within 0.05; W0's drop over M's windows. Then A per state, P2, the branch.
# Every scored number is printed to this log.
# Inputs on NFS (gitignored): experiments/keydependent_elicit/results/large/{windows_elicit,windows_cipher,
# {elicit,cipher,cipher_own}_s{0,1}}.pt; experiments/keydependent_patch/results/large/{elicit,cipher}_s{2,3,4}.pt
# (~113 MB each); Pile rows artifacts/sets/train_32768_32768.npy (cached, E59); the target model and decomposition in
# artifacts/vpd/; E54's stage1.json, stage2.json and E56's job2.json (committed).
# Writes experiments/readout64/results/run.json (small) and results/large/readouts.pt (< 1 MB). Commits nothing.
# Expected: ~8 min. Model load ~1, inputs ~2 (E59, job 48412), caching ~65K windows ~1-2 (from E59's 0.233 s per 96 windows,
# an upper bound: it included training), training < 1, possession and controls ~1, removals ~2. 30-min wall.
#SBATCH --job-name=commute-readout64
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/readout64/run.py "$@"
