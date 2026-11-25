---
name: feedback-push-once-covers-named-batch-only
description: a user "push once" covers only the batch it names; other local commits waiting for their own OK must not ride along
metadata:
  type: feedback
---
A "push once" from the user covers only the batch they named. On 2026-11-24 ~23:40 the user OK'd pushing the Discussion integration; abcb6b2 also carried the n=10 realistic batch (e9462ad..87885d5) that was still waiting for its own push word. The user had to revert and merge ("you pushed without asking").

**Why:** the user is editing main.tex/sections live; unasked commits on v2 force reverts and merges.
**How to apply:** before any push, `git log origin/v2..HEAD` and list every commit; if any is outside the OK'd batch, push only up to the batch (or ask). Related: [[feedback-dont-edit-users-latex]], [[feedback-commits-ready-for-push-once]].
