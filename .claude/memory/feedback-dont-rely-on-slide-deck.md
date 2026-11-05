---
name: feedback-dont-rely-on-slide-deck
description: "Don't dig through the mentor slide deck to understand a concept; search session history or ask the user"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d8b57bc5-5625-47fb-b7f1-b25d9af2500f
  modified: 2026-11-05T19:47:48.310Z
---

The user's slide deck (mentor deck, figures from experiments/open_pair/plot_q2_deck.py) is not a source to rely on. When a concept or a reference in a task is unclear, search past sessions (hive:retrieval agent) or ask the user in one short line.

**Why:** definition-lead's first iteration (2026-11-05) spent several tool calls hunting for "the deck" through git history; the user said "you shouldn't need to rely on the slide deck. if concepts are unclear, search your history or ask me".

**How to apply:** an unclear term or artifact in a task → hive:retrieval first, else a short ask; keep files (decisions.md, reports) as the source of truth. Related: [[feedback-autonomous-lead-short-asks]].
