#!/usr/bin/env bash
# Stop hook: block the session's end once if the record changed after the research log did.
# Exit 2 + stderr = Claude is told to update docs/research_log/index.html (and republish) before stopping.
set -u
input=$(cat)
if printf '%s' "$input" | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("stop_hook_active") else 1)' 2>/dev/null; then
  exit 0
fi
cd "$(dirname "$0")/../.." || exit 0
log=docs/research_log/index.html
[ -f "$log" ] || exit 0
newer=$(find experiments/TRIED.md decisions.md GOALS.md reports -maxdepth 1 \
  \( -name TRIED.md -o -name decisions.md -o -name GOALS.md -o -name '*_analysis_*.md' -o -name '*_decision_*.md' \) \
  -newer "$log" 2>/dev/null | sort)
[ -z "$newer" ] && exit 0
{
  echo "Research log check: these files changed after docs/research_log/index.html was last updated:"
  echo "$newer" | sed 's/^/  - /'
  echo "Update the log (experiment card, claims table, sub-question status, timeline, changelog row), republish it as docs/research_log/README.md says, then stop. If nothing in those files changes what the log says, say so in one line and run: touch docs/research_log/index.html"
} >&2
exit 2
