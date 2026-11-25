---
name: feedback-commits-ready-for-push-once
description: "Keep paper edits committed locally and built while pushes are held, so \"push once\" is one command"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 9f6fbfa5-7835-4b99-a63b-db18661b517a
  modified: 2026-11-25T00:19:09.894Z
---

While pushes or edits are on hold, keep the batch applied as local commits, built clean, so that when the user says "push once" it is pull + push right away.

**Why:** 2026-11-24 the user said "push once"; nothing was committed (the batch sat as drafts behind the HOLD), so I asked a question and then applied the batch, and the user was left waiting: "in future, please have commits ready so when i say 'push once' you can just do it".

**How to apply:** under an edit HOLD, prepare the batch in a separate worktree/branch (not the shared checkout the user edits) and rebase/merge it onto their pushes when they finish; under a push-only hold, commit locally as items land. User-owned lines (main.tex, sec7, abstract) still go as old/new first. See [[feedback-dont-edit-users-latex]], [[feedback-avoid-paper-merge-conflicts]].
