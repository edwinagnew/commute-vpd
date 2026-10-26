#!/usr/bin/env bash
# List agents that have waited at a permission or trust prompt, or at a question, for more than BLOCKED_MIN minutes
# (from mats-fleet's ci/check_blocked.sh). Run by cron every 5 minutes. Writes FLEET_DATA/blocked.txt (empty when
# nothing is blocked); fleet/iteration-start shows it to the lead, which sends a notification and puts it at the top
# of the morning digest. State: when each window was first seen blocked, in FLEET_DATA/logs/blocked_since/.
set -uo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/fleet.sh"
BLOCKED_MIN=${BLOCKED_MIN:-10}
state="$FLEET_DATA/logs/blocked_since"
out="$FLEET_DATA/blocked.txt"
mkdir -p "$state"
now=$(date +%s)
report=""
tmux has-session -t "$FLEET_SESSION" 2>/dev/null || { : > "$out"; exit 0; }
for w in $(tmux list-windows -t "$FLEET_SESSION" -F '#W'); do
  tail=$(tmux capture-pane -p -t "$FLEET_SESSION:$w" 2>/dev/null | grep -v '^\s*$' | tail -25)
  if grep -qE 'Do you want to (proceed|make this edit|create)|Allow Claude to|Allow this read|trust this folder|Yes, and don.t ask again|❯ 1\. Yes|Enter to select' <<< "$tail"; then
    [[ -f "$state/$w" ]] || echo "$now" > "$state/$w"
    since=$(cat "$state/$w"); mins=$(( (now - since) / 60 ))
    if (( mins >= BLOCKED_MIN )); then
      kind="a permission prompt"; grep -q 'Enter to select' <<< "$tail" && kind="a question"
      what=$(grep -m1 -oE '(Bash|Edit|Write)\([^)]*\)|Allow Claude to run [^?]*' <<< "$tail" | head -1)
      report+="$(date -d @"$since" '+%Y-%m-%d %H:%M') $w waiting at $kind for ${mins} min${what:+: $what}"$'\n'
    fi
  else
    rm -f "$state/$w"
  fi
done
printf '%s' "$report" > "$out"
