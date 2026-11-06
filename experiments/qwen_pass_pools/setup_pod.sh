#!/usr/bin/env bash
# E120's pod set-up (one RunPod H100), from the synced repo root, once after the first sync():
#
#   bash experiments/qwen_pass_pools/setup_pod.sh
#
# Builds vLLM 0.11.0's own venv at /root/vllm-venv (never the project's .venv) from pod-requirements.txt: every
# package pinned, resolved by `uv pip compile pod-requirements.in --python-version 3.12 --python-platform
# x86_64-manylinux_2_28` (vllm 0.11.0, torch 2.8.0, transformers 4.57.3, datasets 4.4.2, math-verify 0.9.0 with
# antlr4 4.13.2). Then fetches Qwen3-1.7B at the repo's pinned revision (`commute.qwen_eval.MODEL_REV`) into HF_HOME
# alongside the venv build. Each step prints "[setup HH:MM:SS +Ns] ..." to /root/setup.log and the RunPod console.
set -Eeuo pipefail
UV_VERSION=0.12.9
VENV=/root/vllm-venv
export HF_HOME=/root/hf
exec > >(tee -a /root/setup.log /proc/1/fd/1) 2>&1
t0=$SECONDS
step() { echo "[setup $(date -u +%T) +$((SECONDS - t0))s] $*"; }
trap 'step "FAILED (exit $?) at line $LINENO: $BASH_COMMAND"' ERR

source cluster/pod_env.sh
step "start in $PWD"
setsid nohup bash cluster/pod_idle_watchdog.sh > /dev/null 2>&1 < /dev/null &
step "idle watchdog started"
[[ "$(uv --version 2>/dev/null)" == "uv $UV_VERSION"* ]] || pip install -q --root-user-action=ignore "uv==$UV_VERSION"
uv python install -q 3.12
step "$(uv --version), python 3.12"

REV=$(sed -n 's/^MODEL, MODEL_REV = "Qwen\/Qwen3-1.7B", "\([0-9a-f]*\)"$/\1/p' src/commute/qwen_eval.py)
[[ ${#REV} == 40 ]] || { step "could not read MODEL_REV from src/commute/qwen_eval.py"; exit 1; }
(uv run -q --no-project --python 3.12 --with huggingface_hub==0.36.0 \
   hf download Qwen/Qwen3-1.7B --revision "$REV" > /root/model-fetch.log 2>&1
 step "model Qwen/Qwen3-1.7B@$REV fetched") &
fetch=$!

uv venv -q --python 3.12 "$VENV"
uv pip install -q --python "$VENV/bin/python" -r experiments/qwen_pass_pools/pod-requirements.txt
step "vllm venv built"
wait "$fetch"

mkdir -p experiments/qwen_pass_pools/results/pod_dl
PYTHONPATH=src "$VENV/bin/python" -c 'import torch, vllm, math_verify, transformers, datasets
assert torch.cuda.is_available()
print("torch", torch.__version__, "vllm", vllm.__version__, "transformers", transformers.__version__,
      "datasets", datasets.__version__, torch.cuda.get_device_name())'
step "done; run: PYTHONPATH=src HF_HOME=$HF_HOME $VENV/bin/python experiments/qwen_pass_pools/pod.py"
