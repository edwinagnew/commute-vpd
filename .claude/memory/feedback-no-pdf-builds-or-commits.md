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

Figures too (2026-11-27): the user renders every included plot on the laptop with `uv run paper/figures/src/make_all.py [names]` (whip-lead, a423aab/939097c; PDF only, from data/*.csv). A figure change = edit its CSV or script, commit, tell the user which name to re-run. Fig 1 TikZ (src/fig1_definition_rho.tex) is outside make_all: the user compiles it in TeXstudio, so after a .tex change say "pull, then recompile" — a PDF built before the pull shows the old panel (11-27, rho overlay).
