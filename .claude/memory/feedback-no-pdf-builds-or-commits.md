---
name: feedback-no-pdf-builds-or-commits
description: "Never build or commit paper main.pdf; the user builds it in TeXstudio; gitignored since 2026-11-26, re-track before the final push"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 6e609008-e20e-4975-b125-b1ecb0625afd
  modified: 2026-11-27T03:51:18.621Z
---

Don't build `paper/paper/main.pdf` and never commit it. The user builds it in TeXstudio after each pull, and a committed copy kept causing pull conflicts (user, 2026-11-26). It was untracked and gitignored in paper commit 4ddd81f.

**Why:** cpu-run builds were slow and flaky (friction 1178), and the committed PDF conflicted with the user's local build.
**How to apply:** check edits with a grep for broken refs, or ask the user, and leave the PDF alone. Before the final push of the draft (due 2026-11-29, [[project-draft-deadline-2026-11-29]]), remove the `.gitignore` line, rebuild once and commit main.pdf so the final push carries the real draft. This supersedes the "rebuild main.pdf after merges" part of [[feedback-avoid-paper-merge-conflicts]].
