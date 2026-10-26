# Sourced by the scripts in fleet/: reads fleet/fleet.conf and points Claude Code's scratchpads at /ephemeral.
# shellcheck source=/dev/null
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/fleet.conf"

# Scratch goes on the dev node's local /ephemeral, never /tmp: /tmp shares the node's small root disk with every
# user (mats-fleet, 20 October 2026: 42 GB of torch environments in a Claude scratchpad filled it to 97%).
# /ephemeral is wiped on reboot, so the folder is created when needed.
if [[ -d /ephemeral ]]; then
  FLEET_SCRATCH="/ephemeral/$(id -un)"
  export CLAUDE_CODE_TMPDIR="$FLEET_SCRATCH/claude-tmp"
  mkdir -p "$CLAUDE_CODE_TMPDIR" 2>/dev/null || true
fi

# The transcript of ROLE's running agent: the live Claude process named FLEET_PREFIX-ROLE (fleet/up's -n) gives its
# session id in ~/.claude/sessions/<pid>.json. Not the newest file in the clone's project folder: a person's own
# Claude session in the same clone writes there too, and fleet/check then reported that session (2026-10-26).
fleet_transcript() {  # fleet_transcript ROLE -> path, or nothing when the agent is not running
  local f pid sid=""
  for f in "$HOME"/.claude/sessions/*.json; do
    [[ "$(jq -r .name "$f" 2>/dev/null)" == "$FLEET_PREFIX-$1" ]] || continue
    pid=$(jq -r .pid "$f"); kill -0 "$pid" 2>/dev/null && sid=$(jq -r .sessionId "$f")
  done
  [[ -n "$sid" ]] && echo "$HOME/.claude/projects/$(tr '/.' '--' <<< "$FLEET_AGENTS/$1")/$sid.jsonl"
}
