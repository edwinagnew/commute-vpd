#!/usr/bin/env bash
# Terminates this RunPod pod after POD_IDLE_MIN (default 30) consecutive minutes at 0% GPU utilisation, whatever its
# controlling agent is doing. Started in the background by cluster/pod_setup.sh. On 2026-10-26 an H100 pod finished
# its run in two minutes, then billed ~14 h (~$40) while its runner and the lead sat at a permission prompt; nothing
# but the session budget could stop it. A pod idle for 30 min has either finished (results should be home) or died.
#
# Uses the pod's own runpodctl and pod-scoped API key, which RunPod puts on every pod; pod_setup.sh prints whether
# both are present, so the first pod shows whether this can work. Log: /root/idle-watchdog.log and the RunPod console.
set -uo pipefail
IDLE_MIN=${POD_IDLE_MIN:-30}
say() { echo "[idle-watchdog $(date -u +%T)] $*" | tee -a /root/idle-watchdog.log > /proc/1/fd/1; }
say "armed: terminate after $IDLE_MIN idle minutes"
idle=0
while sleep 60; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  if [[ ${util:-0} -gt 0 ]]; then idle=0; else idle=$((idle + 1)); fi
  (( idle > 0 && idle % 10 == 0 )) && say "GPU idle $idle min"
  if (( idle >= IDLE_MIN )); then
    say "GPU idle $idle min: terminating pod ${RUNPOD_POD_ID:-?}"
    runpodctl remove pod "${RUNPOD_POD_ID:?}" 2>&1 | tee -a /root/idle-watchdog.log > /proc/1/fd/1
    idle=0  # if the call failed, try again after another IDLE_MIN
  fi
done
