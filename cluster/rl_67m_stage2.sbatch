#!/bin/bash
# E220 (stage 2b) replaced E218's KL stop with a size stop and its calib jobs: predictions_E220.json, no calib, outputs
# stage2b_*; its job list and times are in reports/plans/2026-11-19_plan_rl-67m_appendix.md (Stage 2b). The lines below
# are E218's as it ran.
# E218 (cell P2.41 stage 2): the arms at matched KL from W0. Plan: reports/plans/2026-11-19_plan_rl-67m.md (Stage 2).
# Settings and bars: experiments/rl_67m/results/predictions_stage2.json (run_stage2.py refuses unless it is committed
# unchanged). One job per argument list; submit from the clone root, the job name and --time per job as below:
#   sbatch --comment=E218 --time=00:20:00 --mem=40G cluster/rl_67m_stage2.sbatch ctl            # first and alone
#   sbatch --comment=E218 --time=00:20:00 cluster/rl_67m_stage2.sbatch calib 1                  # after ctl; calib 2 alike
#   sbatch --comment=E218 --time=00:45:00 cluster/rl_67m_stage2.sbatch null 3e-05               # after both calibs; 0.0001, 0.0003 alike
#   sbatch --comment=E218 --time=00:20:00 cluster/rl_67m_stage2.sbatch arm 1 rl 0               # after both calibs; seeds 1, 2 and pick 2 alike
#   sbatch --comment=E218 --time=00:30:00 cluster/rl_67m_stage2.sbatch arm 1 own                # seeds 0-2 in one job; demo alike
#   sbatch --comment=E218 --time=00:20:00 cluster/rl_67m_stage2.sbatch arm anchor demo
#   sbatch --comment=E218 --time=00:30:00 cluster/rl_67m_stage2.sbatch arm 1 shuffled 0         # after all three null jobs
# Times on an L40 from E215's measured sampling, 2.77 s per 256 sequences x 16 tokens (so ~1.6 s per RL step of 128
# sequences): model load ~2 min; ctl ~6 (E209's stage 0 ~3); calib ~12 (300 steps + 11 held-out reads of ~11 s);
# rl arm ~11 per seed; shuffled ~19 per seed at most (600 steps); null ~35 at most (two picks x 600 steps; the one job
# kind over 30 min, still inside debug's 2 h cap); own ~4 per seed (1.5 min of W0 samples, then fast supervised steps);
# demo ~3 per seed. Host memory: ctl reads E53's gate sets for the share (~20 GB, as E209), the others under 10 GB.
# Inputs: the model in artifacts/vpd/; artifacts/sets/{val_0_64,val_0_256,train_0_4096,train_0_32768}.npy;
# experiments/rl_67m/results/{stage0,stage1,predictions}.json and results/large/stage1_samples.pt (E215);
# experiments/transport_test/results/large/windows.pt; ctl also E209's inputs (E53's states, E136/E143 records);
# French Wikipedia streamed for the anchor's demonstrations (articles 258 to ~1,300).
# Writes experiments/rl_67m/results/stage2_<job>.json, results/stage2_e53_reread/stage0.json (ctl) and the arms' states
# results/large/stage2/<behaviour>_<arm>_s<seed>.pt (bf16, ~57 MB each, 27 states, ~1.5 GB).
#SBATCH --job-name=commute-rl67m-s2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/rl_67m/results/predictions_E220.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/rl_67m/run_stage2.py "$@"
