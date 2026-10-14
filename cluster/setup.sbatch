#!/bin/bash
# One-off (re)build of the project venv on the GPU node, plus GPU sanity checks.
# Submit from the controller:  cd /mnt/nw/home/e.agnew/commute && sbatch cluster/setup.sbatch
# (uv sync downloads torch; that is heavy work and must not run on the controller.)
#SBATCH --job-name=commute-setup
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/nw/home/e.agnew/commute/cluster/logs/%x-%j.out

set -euo pipefail
source /mnt/nw/home/e.agnew/commute/cluster/env.sh
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log

echo "== host $(hostname)  job $SLURM_JOB_ID  qos $SLURM_JOB_QOS  gpus $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvidia-smi | grep -o "CUDA Version: [0-9.]*" || true

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
uv --version
uv python install 3.13
uv sync --locked                       # default groups include dev (ipykernel, jupyter-server)

.venv/bin/python - <<'PY'
import torch, ipykernel, jupyter_server
print("torch", torch.__version__, "cuda build", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
print("matmul ok", float((x @ x).float().abs().mean()))
print("ipykernel", ipykernel.__version__, "jupyter_server", jupyter_server.__version__)
PY

echo "== WANDB_API_KEY in job env: $([ -n "${WANDB_API_KEY:-}" ] && echo set || echo unset)"
echo "== sshd/pam on node:"
grep -H pam_slurm /etc/pam.d/* 2>/dev/null || echo "no pam_slurm* entry readable in /etc/pam.d"
(command -v ss >/dev/null && ss -ltn | grep -E ':22\s') || echo "nothing listening on :22 (or ss missing)"

PATH="$COMMUTE_REPO/.venv/bin:$PATH" EXPECTED_GPUS=1 EXPECTED_GPU_NAME=L40 PERSISTENT_PATH="$COMMUTE_REPO" \
    bash cluster/gpu-sanity.sh || echo "gpu-sanity exit $?"
echo "== done $(date -Is)"
