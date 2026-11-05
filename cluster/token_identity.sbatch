#!/bin/bash
# E95, one L40, debug QoS, $0: experiments/token_identity/run.py. Rules: experiments/token_identity/results/predictions.json (the script
# refuses unless it is committed unchanged). Plan: reports/plans/2026-11-05_plan_e95-token-identity.md. Submit from the clone root, no arguments:
#   sbatch cluster/token_identity.sbatch
# Time ~5 min, from E90 Amendment 2's job, which did more in 4 min 45 s (cluster/logs/commute-e90-52444.out:1 14:42:41, :118 14:47:26):
# model and rows under 1 min (:7, with a guess scan E95 does not run); two capabilities ~25 s each (:8 to :9); two controls ~16 s a block and
# do-nothing ~20 s (:26, :28); ~16 rows at ~1 s (:79 to :96, two blocks there, one here). Write fits ~1 s each (commute-e90-52363.out:68 to :70),
# eight of them; 80 rotated-table reads on 97 windows, unmeasured, about 1 min at most. Memory: :119 peak 6.1 GB, plus eight 113 MB states kept
# between the stages. Inputs on NFS (gitignored): regating_table/results/large/{windows_,}{ident32,ident32b,rand32,rand32b}*, new_vs_reuse/results/large/bases.pt,
# artifacts/sets/train_32768_32768.npy, artifacts/vpd/. Writes results/run.json (small, commit), results/large/{reads_*,X_*}.pt (~40 MB: 4 tables, 9 MB each).
# Exit 0 done; 1 a stop (reproduction or placement, `stopped` in run.json) or a refusal.
#SBATCH --job-name=commute-e95
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
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/token_identity/run.py
