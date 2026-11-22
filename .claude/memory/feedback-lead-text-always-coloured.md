---
name: feedback-lead-text-always-coloured
description: "Every sentence I write in the paper is green (\\new / revised), on every branch, including restructures; never accept marks to black (user, 2026-11-22, \"very important to always do\")"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 13a81ff1-83e6-47b7-a791-be1c0060a217
  modified: 2026-11-22T22:11:02.239Z
---

On the v2 restructure branch I accepted all marks (\new → plain) and wrote new sections in black, so the user saw "a lot of black text i definitely didnt write". They called it a violation and "very important to always do".

**Why:** the user must be able to tell their own prose from the leads' at a glance; black means "the user wrote this".

**How to apply:** new files get \begin{revised}...\end{revised}; new section titles \section[T]{\new{T}}; floats need \color{green!45!black}\captionsetup{font+=lead,labelfont+=lead} inside (the env colour does not reach floats or captions). Never redefine \new/\old/revised to plain, even on a branch; only \sugg may be hidden. Render a page and check before pushing. Related: [[feedback-dont-edit-users-latex]], [[feedback-paper-condense-claims]].
