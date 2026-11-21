---
name: feedback-whip-lead-autonomy
description: "whip-lead acts on its own — stops/starts/restarts leads, relays results between leads, keeps schedule.md — rather than handing each step to the user"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 239d2663-791e-4483-8c4b-1aa692a90a77
  modified: 2026-11-20T21:57:57.189Z
---

As whip-lead, act autonomously: stop finished or idle leads, restart a drifted lead with a corrected task, start stopped
leads when the plan gives them work, start redteam-lead for the cold read, and pass results between leads (e.g. tell
writeup-lead about a filled cell) without being asked. Gaps in the user's plan go in
`~/commute-data/agents/whip-lead/schedule.md`; the user's plan wins.

**Why:** user, 2026-11-20: "you should be doing this more autonomously, including stopping/starting leads at your
discretion"; they had to run `fleet/stop qwen-lead` and ask me to relay P2.44 themselves.

**How to apply:** report the action in chat afterwards, don't propose it. Ask only for what the harness reserves to
the user (cell proposals, ladder, definition wording) or what changes the plan. Urgent pokes: note-*.md plus SendMessage.
