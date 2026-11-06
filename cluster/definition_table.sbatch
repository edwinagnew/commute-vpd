#!/bin/bash
# E116, the 67M task table under the frozen definition: one job, stages 0 (E112 reproduction alone, then P0 per readout),
# 1 (P1 per state), 2 (per readout: do-nothing first, then P2, P3, twins' P3; the table). Plan:
# reports/plans/2026-11-06_plan_e116-definition-table.md. Rules: experiments/definition_table/results/predictions.json
# (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && ~/commute-agents/fleet/link-data <role> && sbatch cluster/definition_table.sbatch
# Exit 0 done; 1 a stop (stage 0: E112's possession numbers not reproduced within 1e-3); 3 too little time left for the
# next readout in stage 2: resubmit (finished stages and readouts are kept), e.g.
#   ~/commute-agents/fleet/resubmit --code 3 --max 3 cluster/definition_table.sbatch
# Inputs: Pile train rows 32768-65535, 0-4095 and val 0-255 (cached in artifacts/sets), the target model in artifacts/vpd/,
# E112's task.pt, E62's, E58's, E46's and E53's windows and states (sha256 checked against their records).
# Writes experiments/definition_table/results/stage{0,1,2}.json (after every step) and run.out (this log, appended);
# results/large/probes_<C>.pt (10 readouts, up to ~25 MB each) and e112_repro/probes.pt (~3 MB).
# Expected time on a debug L40 (estimate, not measured): model load ~1 min, stage 0 ~6 min (E112 repro ~0.5, Pile scans
# and ten probe fits ~5), stage 1 ~2 min (26 states, ~1,000 windows each), stage 2 ~6 min; ~15 min in all. Memory: the
# probe rows (134 MB), up to ~1 GB of Pile residuals per readout and their float64 copy in the ridge; 24G.
#SBATCH --job-name=commute-e116
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
rc=0
.venv/bin/python experiments/definition_table/run.py 2>&1 | tee -a experiments/definition_table/results/run.out || rc=$?
echo "=== FINISHED E116 (exit $rc) — results in $COMMUTE_REPO/experiments/definition_table/results/ ==="
exit $rc
