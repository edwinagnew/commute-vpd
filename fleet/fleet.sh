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
