---
name: feedback-rename-everywhere
description: A renamed term changes everywhere at once (log, GOALS Terms, CLAUDE.md, README, TRIED, memory) and the old name goes into check.py's RETIRED list; dated reports stay as written
metadata:
  type: feedback
---

When a term is renamed, replace it in every living file in the same change: the research log's parts, `GOALS.md` →
Terms (old name in its "old names" column), `.claude/CLAUDE.md`'s retired list, `README.md`, `experiments/TRIED.md`,
skills/agents and memory. Add the old name to `RETIRED` in `.claude/skills/writeup/check.py`. Dated reports and old
changelog rows stay as written.

**Why:** 2026-10-22 the user renamed H′ to "the fixed-shift hypothesis" and said: "terminology should be consistent
otherwise youll probably revert to using bad names in the log". New sessions copy whatever name the files they read use.

**How to apply:** prefer names that say what the thing is over symbols (H′ meant nothing to an outside reader).
Related: [[feedback-user-definitions-override-files]], [[feedback-define-terms-when-reporting]].
