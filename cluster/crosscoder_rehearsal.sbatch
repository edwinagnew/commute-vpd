#!/bin/bash
# P2.31 run 2 (E203): the $0 rehearsal of every stage at tiny sizes (settings.rehearsal in
# experiments/crosscoder/results/run2/predictions.json: 1,536 latents, 4.1M tokens, 2,000 steps; permissive bars except
# the self control's exact ones). Gate 3 of reports/plans/2026-11-17_plan_p2-31-run2.md. Covers: the pinned
# dictionary-learning import, the trainer on our stream, a checkpoint save (--stop-after, exit 3) and the resume from it,
# the snapshots, latent scaling, the change reading, the timing stage, stage 1 and one stage-2 shard; outputs under
# experiments/crosscoder/results/run2_rehearsal/ and results/large/run2_rehearsal/ (cleared first).
#   cd ~/commute-agents/compare-lead && ~/commute-agents/fleet/link-data compare-lead && sbatch --comment=E203 cluster/crosscoder_rehearsal.sbatch
# Inputs on NFS (gitignored): artifacts/vpd/, artifacts/sets/{train_0_4096,val_0_64}.npy, the states'
# results/large/ named in experiments/share_untrained/results/inventory.json and refit_96key's, and their task windows.
# Time: not measured; ~15 small crosscoders plus model loads, estimated 30-60 min on an L40.
#SBATCH --job-name=commute-crosscoder-rehearsal
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -L experiments/refit_96key/results/large || { echo "results/large is not linked: run ~/commute-agents/fleet/link-data compare-lead"; exit 1; }
rm -rf experiments/crosscoder/results/run2_rehearsal experiments/crosscoder/results/large/run2_rehearsal
RUN=(.venv/bin/python experiments/crosscoder/run.py)

echo "== save: the self control stops after step 1000 (exit 3 expected)"
rc=0; "${RUN[@]}" 0a --rehearse --stop-after 1000 || rc=$?
[ "$rc" -eq 3 ] || { echo "REHEARSAL FAIL: --stop-after exited $rc, not 3"; exit 1; }
ls -la experiments/crosscoder/results/large/run2_rehearsal/ckpt/
echo "== resume: the self control from its checkpoint"
"${RUN[@]}" 0a --rehearse
grep -q '"pass_a": true' experiments/crosscoder/results/run2_rehearsal/stage0.json || { echo "REHEARSAL FAIL: no pass_a"; exit 1; }
echo "== planted, timing, stage 1, one stage-2 shard"
"${RUN[@]}" 0b --rehearse
"${RUN[@]}" t --rehearse
"${RUN[@]}" 1 --rehearse
"${RUN[@]}" 2 0/40 --rehearse
ls -la experiments/crosscoder/results/run2_rehearsal/sides/ experiments/crosscoder/results/large/run2_rehearsal/
echo "REHEARSAL DONE"
