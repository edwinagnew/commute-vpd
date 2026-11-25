#!/bin/bash
# Lift test: LoRA arm (ranks 1, 8 on h.2.mlp.down_proj, arm A data), then stage 0 on its checkpoints.
# Submit from the controller:  cd ~/commute-agents/<role> && sbatch cluster/lift_lora.sbatch
#SBATCH --job-name=commute-lift-lora
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
R=experiments/lift_test/results
.venv/bin/python experiments/lift_test/run.py lora
mv $R/obstruction.json $R/obstruction_round1.json
.venv/bin/python experiments/lift_test/run.py obstruction $R/large/LoRA_r*/dW_step*.pt
mv $R/obstruction.json $R/obstruction_lora.json && mv $R/obstruction_round1.json $R/obstruction.json
