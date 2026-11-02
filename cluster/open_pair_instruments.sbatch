#!/bin/bash
# E55 Stage C3: pruning revival, reverse fine-tuning (toward W0's distribution) and the possession probe on E58's four
# 67M key -> value maps, eight seeds each. Plan: reports/plans/2026-10-31_plan_e55-stage-c-open-pair.md. Rules:
# experiments/open_pair_instruments/results/predictions.json (run.py refuses unless it is committed unchanged; filing A
# for measure, filing B as well for seeds and instruments). Code: experiments/open_pair_instruments/{run,measure,seeds,
# instruments,common}.py and src/commute/revival_kv.py.
#   cd ~/commute-agents/<role>
#   sbatch --job-name=commute-e55-c3-measure cluster/open_pair_instruments.sbatch measure          # job 1, ~8-10 min (estimated)
#   (append filing B to predictions.json from results/measure.json's filing_B_draft and commit it; jobs 2-3 exit 2 until then)
#   sbatch --job-name=commute-e55-c3-seeds cluster/open_pair_instruments.sbatch seeds              # job 2, ~12-18 min (estimated)
#   sbatch --job-name=commute-e55-c3-instruments cluster/open_pair_instruments.sbatch instruments  # job 3, ~15-25 min (estimated)
#   (job 4 = the same instruments line again after an exit 3; seeds also resumes the same way)
#   sbatch --job-name=commute-e55-c3-ladder cluster/open_pair_instruments.sbatch ladder            # step ladder, 1-3 jobs
#   (plan reports/plans/2026-10-31_plan_e55-c3-ladder.md; needs results/predictions_ladder.json committed; resubmit on
#   exit 3. Part A: C3's 32 states at K up to 320, ~3 min; part B: 9 fine-tunes x 4 steps -> results/ladder.json)
# measure: stop rows (sha256 of the 12 states and 8 windows files, identical tokens, G within 1e-4, zero update D = 0),
#   D / off-answer KL / MLP share per state, the prior per map, the random-removal spread on the six pole states, the
#   planted wrapper built and checked -> results/measure.json (refuses if it exists). No deadline exit (one pass).
# seeds: seed-0 reproduction of each map first (exit 1 on a miss), then seeds 3-7 -> results/seeds.json and
#   results/large/<map>_s<seed>.pt (~113 MB each, 2.3 GB); host RAM peak ~5 GB (a cipher fine-tune's 32 checkpoints).
# instruments: stage 0 (zero update, lens, planted pruning and reverse, weights back at W0), then the 32 states ->
#   results/run.json after each, then q and the scores.
# Exit codes: 0 done; 1 a stop row (tell the user); 2 run out of order (a filing or job 2 missing, or results written
#   under other predictions); 3 deadline (25 min): saved, not finished. On exit 3 resubmit the same command by hand
#   (compute nodes have no sbatch, so a job cannot resubmit itself); the next job resumes from the saved files.
#SBATCH --job-name=commute-e55-c3
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; puts this clone's src first on PYTHONPATH
cd "$COMMUTE_REPO"
stage=${1:-}
case "$stage" in
  measure | seeds | instruments | ladder) ;;
  *) echo "usage: sbatch cluster/open_pair_instruments.sbatch {measure,seeds,instruments,ladder}" >&2; exit 2 ;;
esac
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
large=experiments/open_pair_instruments/results/large
mkdir -p "$HOME/commute-data/results-large/open_pair_instruments"   # also mends a dangling link's target
[ -L "$large" ] || [ -d "$large" ] || ln -s "$HOME/commute-data/results-large/open_pair_instruments" "$large"
rc=0
.venv/bin/python experiments/open_pair_instruments/run.py --stage "$stage" --deadline-min 25 2>&1 | tee -a "$large/$stage.log" || rc=$?
if [ "$rc" = 3 ]; then
  echo "=== DEADLINE (exit 3): saved, not finished; resume by hand: sbatch --job-name=$SLURM_JOB_NAME cluster/open_pair_instruments.sbatch $stage ==="
  exit 3
fi
echo "=== FINISHED E55 C3 $stage (exit $rc) — results in $COMMUTE_REPO/experiments/open_pair_instruments/results/ ==="
exit $rc
