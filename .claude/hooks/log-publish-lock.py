#!/usr/bin/env python3
"""PreToolUse hook on the Artifact tool: a research-log publish needs fleet/log-lock held by this session.

Two sessions published v177 and v178 while a clerk held the lock, forcing two merge rounds and a renumber (friction
347, 149); README step 0 was a rule nothing enforced, and log-lock cannot gate the publisher itself. This hook can: a
publish to the log's artifact (its url, or a page from docs/research_log/build/) is refused unless the lock file
(FLEET_DATA/log-lock: role, taken, pid, session id, note) names this session. Subagents share their session's id, so a
clerk's take covers the session that sent it, as in log-lock's own check. A holder with no session on record (taken
from a plain shell) cannot be told apart and passes, as log-lock's is_mine does; so does a machine without the lock
directory. Reads, lists and other artifacts are never refused.
"""
import json
import os
import sys
import time
from pathlib import Path

LOG_ID = "SvgkzUiESRkyE9GM65HCWu"

try:
    call = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
if call.get("tool_name") != "Artifact":
    sys.exit(0)
args = call.get("tool_input") or {}
if (args.get("action") or "publish") != "publish":
    sys.exit(0)
target = f"{args.get('url') or ''} {args.get('file_path') or ''}"
if LOG_ID not in target and "docs/research_log/build/" not in target:
    sys.exit(0)

data = Path(os.environ.get("FLEET_DATA") or Path.home() / "commute-data")
if not data.is_dir():
    sys.exit(0)
try:
    fields = (data / "log-lock").read_text().split()
except OSError:
    fields = []
me = call.get("session_id") or ""
take = "Run ~/commute-agents/fleet/log-lock take <role> --wait (README.md, How to publish, step 0), pull, rebuild, then publish."
if len(fields) < 4:
    print(f"Research-log publish refused: nobody holds fleet/log-lock (friction 347). {take}", file=sys.stderr)
    sys.exit(2)
role, taken, _pid, sid = fields[:4]
if sid in ("", "-") or not me or sid == me:
    sys.exit(0)
try:
    since = f", taken {int((time.time() - int(taken)) / 60)} min ago"
except ValueError:
    since = ""
print(f"Research-log publish refused: fleet/log-lock is held by {role} in another session ({sid[:8]}{since}; you are "
      f"{me[:8]}), friction 347. Wait for it, then: {take} `log-lock status` says whether the holder is still at work.",
      file=sys.stderr)
sys.exit(2)
