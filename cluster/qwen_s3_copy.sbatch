#!/bin/bash
# Copy the converted Qwen3-1.7B decomposition (53.65 GB, $COMMUTE_QWEN_DIR) onto the RunPod network volume ydavm7ct4x
# (US-MO-1) through RunPod's S3 API, so pods in US-MO-1 mount it at /workspace instead of fetching it from W&B.
#   cd ~/commute-agents/<role>
#   sbatch cluster/qwen_s3_copy.sbatch   # needs RUNPOD_S3_API_USER and RUNPOD_S3_API_KEY in the submitting shell
# Output: the log ends with one "ok"/"MISMATCH" line per file (local vs volume size) and "=== FINISHED (exit 0) ===".
#SBATCH --job-name=commute-qwen_s3_copy
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00   # ~1 h at the W&B upload's 19 MB/s; a short limit backfills sooner
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
trap 'echo "=== FINISHED (exit $?) ==="' EXIT
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
test -n "${RUNPOD_S3_API_USER:-}" || { echo "RUNPOD_S3_API_USER is not set"; exit 1; }
test -n "${RUNPOD_S3_API_KEY:-}" || { echo "RUNPOD_S3_API_KEY is not set"; exit 1; }

# The RunPod names map to the AWS ones inside this job only; the config file lives on the node's /ephemeral.
export AWS_ACCESS_KEY_ID=$RUNPOD_S3_API_USER AWS_SECRET_ACCESS_KEY=$RUNPOD_S3_API_KEY AWS_DEFAULT_REGION=us-mo-1
export AWS_CONFIG_FILE=$TMPDIR/aws-config-$SLURM_JOB_ID
printf '[default]\ns3 =\n  multipart_chunksize = 64MB\n  max_concurrent_requests = 4\n' > "$AWS_CONFIG_FILE"
aws() { uvx --python 3.12 --from awscli==1.36.0 aws --endpoint-url https://s3api-us-mo-1.runpod.io "$@"; }
dest=s3://ydavm7ct4x/qwen3-1.7b-p-e9fcffd5-torch

start=$(date +%s)
aws s3 sync --no-progress "$COMMUTE_QWEN_DIR" "$dest"
echo "[s3] synced in $(( ($(date +%s) - start) / 60 )) min"

# Verify: every local file is on the volume with the same byte count.
listing=$(aws s3 ls --recursive "$dest/")
status=0
for f in "$COMMUTE_QWEN_DIR"/*; do
    name=$(basename "$f"); local_size=$(stat -c %s "$f")
    remote_size=$(awk -v n="qwen3-1.7b-p-e9fcffd5-torch/$name" '$4 == n {print $3}' <<< "$listing")
    if [ "$remote_size" = "$local_size" ]; then echo "ok $name $local_size"; else echo "MISMATCH $name local=$local_size volume=${remote_size:-absent}"; status=1; fi
done
exit $status
