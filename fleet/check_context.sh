#!/usr/bin/env bash
# Restart agents whose context has grown past CONTEXT_LIMIT tokens, in two steps (from mats-fleet's
# ci/check_context.sh). Run by cron every 5 minutes.
#  1. When an agent's last request carried more than CONTEXT_LIMIT tokens, write FLEET_DATA/agents/<role>/
#     restart-requested. The agent sees it in fleet/iteration-start, commits and pushes, updates its handoff note and
#     touches restart-ready, but only when no pod-runner subagent is live and no pod is unwatched: a restart kills
#     background subagents, and pods with cleanup = "terminate" die with their session (2026-10-26).
#  2. When restart-ready exists, close the window and start it again with fleet/up.
# Context size comes from the running agent's own transcript (fleet_transcript in fleet/fleet.sh). Log: FLEET_DATA/logs/context.log
set -uo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/fleet.sh"
CONTEXT_LIMIT=${CONTEXT_LIMIT:-300000}
here=$(dirname "$(readlink -f "$0")")
log="$FLEET_DATA/logs/context.log"
mkdir -p "$(dirname "$log")"
tmux has-session -t "$FLEET_SESSION" 2>/dev/null || exit 0
stamp() { date '+%Y-%m-%d %H:%M %Z'; }

for role in $(tmux list-windows -t "$FLEET_SESSION" -F '#W'); do
  dir="$FLEET_DATA/agents/$role"; mkdir -p "$dir"
  [[ -f "$dir/done" ]] && continue   # finished (fleet/finish): idle until the user gives it new work
  transcript=$(fleet_transcript "$role")
  tokens=0
  [[ -n "$transcript" ]] && tokens=$(tail -n 200 "$transcript" | jq -r 'select(.message.usage?) | .message.usage
      | (.input_tokens // 0) + (.cache_read_input_tokens // 0) + (.cache_creation_input_tokens // 0)' 2>/dev/null | tail -1)
  tokens=${tokens:-0}
  if [[ -f "$dir/restart-ready" ]]; then
    # Recreate the window, so the restart also picks up changes to fleet/up.
    tmux kill-window -t "$FLEET_SESSION:$role" 2>/dev/null
    sleep 2
    "$here/up" "$role" >>"$log" 2>&1 && echo "$(stamp) $role: restarted at $tokens tokens" >>"$log"
    rm -f "$dir/restart-ready" "$dir/restart-requested"
    date +%s >"$dir/restarted-at"
  elif (( tokens > CONTEXT_LIMIT )) && [[ ! -f "$dir/restart-requested" ]]; then
    # Until the fresh session sends its first request, the newest transcript may still be the old session's.
    last=$(cat "$dir/restarted-at" 2>/dev/null || echo 0)
    (( $(date +%s) - last < 900 )) && continue
    echo "context $tokens tokens, limit $CONTEXT_LIMIT; requested $(stamp)" >"$dir/restart-requested"
    echo "$(stamp) $role: restart requested at $tokens tokens" >>"$log"
  fi
done
