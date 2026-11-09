#!/bin/bash
# E137 (cell P1.3): the first test of the second wording of reuse on four pairs whose label is known. Plan:
# reports/plans/2026-11-09_plan_reuse-variable-first-test.md (apparatus in its appendix, section D). Rules: experiments/reuse_variable_test/results/predictions.json
# (the run refuses unless it is committed unchanged). One L40, debug QoS, $0. Submit from the clone root, no arguments:
#   cd ~/commute-agents/definition-lead && sbatch cluster/reuse_variable_test.sbatch
# Stages write results/stage{0,1,2,3}.json, each printing `STAGE n DONE` after its file is written; the runner waits on that line. Stage 0 stops the
# job (exit 1) if M' at start is not M. Exit 3 means stage 2 reached the wall limit (105 min) with its partial results saved: resubmit this script, it resumes.
# Inputs on NFS (gitignored): artifacts/vpd/ (the model), artifacts/sets/train_0_32768.npy, train_32768_8192.npy, val_0_2000.npy.
# Writes: results/stage*.json, results/rules.json, results/large/pieces.pt (the trained pieces, ~2.5 MB, gitignored).
# Time, untimed on a GPU (the code has not run on one; a CPU stand-in reached stage 1 only, and the scored run is its own smoke): stage 0 ~15 min (gaps, rank from eigenvalues of X'X (30,720 square, float64), the capacity ridge), stage 1 ~3 min,
# stage 2 ~30 min (144 G searches at 400 steps, H ridge and refinement per pair), stage 3 ~1 min. One or two debug jobs.
# Memory, by arithmetic and not measured: at the `final` site the card holds the 36,864-column X'X in float64 (10.9 GB), a copy and its Cholesky (10.9 GB each) and the validation inputs (2.9 GB), about 36 GB of the L40's 48 GB; the host holds the Pile rows (~200 MB) and the lowercased copies (~10 MB), so 32G is ample.
#SBATCH --job-name=commute-e137
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:55:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --comment=E137

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/reuse_variable_test/run.py
