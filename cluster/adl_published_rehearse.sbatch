#!/bin/bash
# P2.67 (E284) rehearsal, $0: the authors' toolkit at 617d9266 on one organism and variant (cake_bake default) at tiny
# sizes (256 texts for delta_bar, 8 eval documents per eval set; 64 + 64 random vectors and positions 0-8 at full size),
# printing the first causal-effect batch's time, then the tar of its outputs. Usage, from the clone root:
#   sbatch --comment='E284 rehearse: toolkit cake_bake default, tiny sample counts' cluster/adl_published_rehearse.sbatch
# Then: ~/commute-agents/fleet/cpu-run python3 experiments/adl_published/control_check.py ... is NOT run on it (one
# organism); read results/large/rehearse/ by hand (the files and keys control_check.py reads must exist).
#SBATCH --job-name=commute-adl-pub-rehearse
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
W=/ephemeral/$USER/adl_p267
OUT=experiments/adl_published/results/large/rehearse
mkdir -p "$W" "$OUT"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
df -h /ephemeral | tail -1
D=experiments/adl_published/pod_control.py
t0=$(date +%s)
python3 $D install --dir "$W/dt" || exit 1
echo "install took $(( $(date +%s) - t0 )) s; venv size $(du -sh "$W/dt/.venv" | cut -f1)"
python3 $D checks --dir "$W/dt" --out "$W/out"
python3 $D run --dir "$W/dt" --out "$W/out" --runs cake_bake:default --rehearse
code=$?
cp "$W"/out/checks.json "$W"/out/*.tar.gz "$W"/out/*.sha256 "$OUT"/
( cd "$OUT" && sha256sum -c cake_bake_default.sha256 )
du -sh "$W/out" "$HF_HOME"
find "$W/out/cake_bake_default/storage/diffing_results" -name results.json | head -30
echo "=== FINISHED (exit $code) — outputs in $COMMUTE_REPO/$OUT ==="
exit $code
