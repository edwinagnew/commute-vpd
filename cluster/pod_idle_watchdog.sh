#!/usr/bin/env bash
# Terminates this pod (RunPod or Vast) after POD_IDLE_MIN (default 30) consecutive minutes at 0% GPU utilisation, whatever its
# controlling agent is doing. Started in the background by cluster/pod_setup.sh. On 2026-10-26 an H100 pod finished
# its run in two minutes, then billed ~14 h (~$40) while its runner and the lead sat at a permission prompt; nothing
# but the session budget could stop it. A pod idle for 30 min has either finished (results should be home) or died.
#
# Uses the pod's own scoped key: runpodctl and RUNPOD_API_KEY on RunPod, CONTAINER_API_KEY on Vast. pod_setup.sh prints
# whether they are present, so the first pod shows whether this can work. Log: /root/idle-watchdog.log and the RunPod console.
#
# One watchdog at a time: starting this script (pod_setup.sh does, or by hand to re-arm with a new POD_IDLE_MIN) ends
# the one named in /root/idle-watchdog.pid. `touch /root/idle-watchdog.stop` disarms it within a minute. Killing
# pod_setup.sh does not stop it (it runs in its own session), and `pkill -f pod_setup.sh` is never the way: it also
# kills the runner's shell (friction 863).
set -uo pipefail
IDLE_MIN=${POD_IDLE_MIN:-30}
PIDF=/root/idle-watchdog.pid STOPF=/root/idle-watchdog.stop
say() { echo "[idle-watchdog $(date -u +%T)] $*" | tee -a /root/idle-watchdog.log > /proc/1/fd/1; }
if old=$(cat "$PIDF" 2> /dev/null) && [[ "$old" != "$$" ]] && grep -qa pod_idle_watchdog "/proc/$old/cmdline" 2> /dev/null; then
  kill "$old" 2> /dev/null && say "replaced the watchdog $old"
fi
echo $$ > "$PIDF"
rm -f "$STOPF"
say "armed: terminate after $IDLE_MIN idle minutes (disarm: touch $STOPF)"
idle=0
while sleep 60; do
  if [[ -e "$STOPF" ]]; then say "disarmed ($STOPF)"; rm -f "$PIDF"; exit 0; fi
  [[ "$(cat "$PIDF" 2> /dev/null)" == "$$" ]] || exit 0  # replaced by a newer watchdog
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  if [[ ${util:-0} -gt 0 ]]; then idle=0; else idle=$((idle + 1)); fi
  (( idle > 0 && idle % 10 == 0 )) && say "GPU idle $idle min"
  if (( idle >= IDLE_MIN )); then
    say "GPU idle $idle min: terminating ${RUNPOD_POD_ID:-${CONTAINER_ID:-?}}"
    if [[ -n "${CONTAINER_ID:-}" ]]; then  # Vast: the instance's own key; tested 2026-11-02
      curl -s -m 30 -K - -X DELETE "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" \
        <<< "header = \"Authorization: Bearer ${CONTAINER_API_KEY:?}\"" 2>&1 | tee -a /root/idle-watchdog.log > /proc/1/fd/1
    else
      runpodctl remove pod "${RUNPOD_POD_ID:?}" 2>&1 | tee -a /root/idle-watchdog.log > /proc/1/fd/1
    fi
    idle=0  # if the call failed, try again after another IDLE_MIN
  fi
done
