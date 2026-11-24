---
name: feedback-avoid-paper-merge-conflicts
description: "Paper repo: pull right before editing and before every commit; the user hit merge conflicts with lead pushes (2026-11-23)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0e36e498-02c3-471f-b62b-3d8a740ab645
  modified: 2026-11-23T22:17:36.576Z
---

The user pushed §4 edits on 2026-11-23 and hit merge conflicts with my earlier pushes: "please try to avoid that in future and make sure your changes are all there".

**Why:** the user edits the same paper files (sec4_setup.tex, main.tex, main.pdf) on their own clone; a lead's push between their pull and their push forces them to merge.

**How to apply:** in the paper repo (`paper/`, branch v2), `git pull --no-rebase` immediately before starting an edit and again before committing; keep each push small and prompt, so it never sits on stale text; never commit main.pdf on its own (it always conflicts, so rebuild it after a merge). After any merge, diff against origin to confirm both sides' changes survived. Pushes still need the user's OK while a push freeze is on (see [[feedback-dont-edit-users-latex]]).
