#!/bin/bash
# P3.5 run 2: experiments/qwen_pair_library/library.py, one arm and seed per job, one L40, $0, for when the cluster can
# start the arms at submission (pods are the default: plan "Run 2 design"). Arguments are passed through; the
# submitter adds --comment. From the clone root:
#   sbatch --comment='E### library: copy seed 0' cluster/qwen_pair_library_run2.sbatch --arm copy --seed 0
# (arms copy cipher known invented template template_shuf gold gold_shuf; seeds 0 1 2). No --upload here: states are
# written straight to results/large/library/ (the symlink into ~/commute-data/results-large/qwen_pair_library).
# Memory and time on an L40 (44 GB): E78's template run peaked 33.7 GB GPU, 13.2 GB host (facts file section 1); here
# one fp32 δW of 5.64 GB is held on the host per read step (the bracket offer) plus one being written, so 32G host.
# Steps: run 1's rehearsal ran 9.5 s/step on an H100 for the short copy rows; E61's template and gold ran 16-17 s/step
# on an L40 (1629 s, 1732 s for 100 steps) with a sparser KL read. Estimated on an L40: built arms 15-45 min, template
# arms ~1.5 h (100 steps plus about five 256-answer generations), gold arms ~1 h. normal QoS, 3 h.
# Log lines to wait on: "STAGE control DONE", "STAGE train DONE", "STAGE read DONE". Exit 1: a stop
# (results/library/stopped_<arm>_s<seed>.json) or a refusal.
#SBATCH --job-name=commute-qwen_pair_library_run2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub   # the model snapshot is here already
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_pair_library/library.py "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
