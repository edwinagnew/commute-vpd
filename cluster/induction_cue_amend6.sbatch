#!/bin/bash
# E62 Amendment 6: W0 only, no training, no state read. Under W0's mask, with B in the text replaced by another token, does the forward lean to B or to the new token?
# Plan: reports/plans/2026-11-02_plan_e62-copying-heads.md, "Amendment 6 (2026-11-02)". Rules: experiments/induction_cue/results/predictions_amend6.json
# (amend6.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/induction_cue_amend6.sbatch
# Exit 1 at: the overwrite not as filed (overwrite_not_as_filed); stage a's control, every mask at 1 then the target model (control_failed); a first cell of the control or the mask off Amendment 5's level by over 1e-3
# (reread_failed); the mask's cells not finite or its lean under 1.0 (unreadable). The fp32 row stops nothing; two precisions disagreeing, or its first cell off Amendment 5's by over 1e-3, is a LOUD line.
# The own-mask reads and the top-1 rates are report-only.
# Inputs (read): experiments/induction_cue/results/{run,amend5}.json, results/large/{windows,heads,amend5_windows}.pt (windows' sha256 checked against run.json, amend5_windows' levels against amend5.json),
# experiments/transport_test/results/large/alive.pt, the VPD artifacts.
# Writes experiments/induction_cue/results/amend6.json after each stage (refuses if it exists), results/large/amend6.out, results/large/amend6_windows.pt (every read per window, 18 x 200 float64 and the top tokens, under 100 KB).
# Deletes nothing, commits nothing.
# Time: expected about 1 min. Amendment 5's job took 59 s in all for about 170 reads (job 49072); this is 18 reads, three top-1 forwards and three Γ passes after the same load of about 30 s. The limit is 10 min, for a queue slot and to catch a hang.
#SBATCH --job-name=commute-e62-amend6
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
large=experiments/induction_cue/results/large
if [ -e "$large/amend6.out" ]; then echo "$large/amend6.out exists: move it aside, this run deletes nothing"; exit 1; fi
rc=0
.venv/bin/python experiments/induction_cue/amend6.py "$@" 2>&1 | tee "$large/amend6.out" || rc=$?
echo "=== FINISHED E62 Amendment 6 (exit $rc) — results in $COMMUTE_REPO/experiments/induction_cue/results/amend6.json ==="
exit $rc
