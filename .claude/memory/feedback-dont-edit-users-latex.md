---
name: feedback-dont-edit-users-latex
description: "writeup-lead may edit writeup/paper/main.tex, but in batches and only pushes it after asking the user; other files push freely"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 6facdd04-e629-41c6-9cbe-e0b92e3dfba0
  modified: 2026-11-17T22:43:16.278Z
---

I may edit `main.tex` (since 2026-11-17 in its own repo, github.com/edwinagnew/commute-vpd-writeup, cloned at `paper/` in the writeup-lead clone: `paper/paper/main.tex`), but only in batches. Before pushing a main.tex change, ask the user in chat. Other files can still be pushed without asking (user, 2026-11-17).

**Why:** The user writes main.tex in TeXstudio. A single unannounced push to it on 2026-11-17 made their push harder.

**How to apply:**
- Collect main.tex edits until there's a batch worth proposing. Until then, keep the items in `paper/paper/lead_notes.md` (the paper repo).
- To propose a batch, commit it locally without pushing and list the changes in chat in a few lines. Push it only after the user says yes, and pull first.
- Don't raise other leads' implementation choices (e.g. crosscoder options). Report their results only.
