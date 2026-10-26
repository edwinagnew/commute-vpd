---
name: feedback-verify-before-announcing-a-cause
description: Code edits to measurement paths must be verified to have landed and run; never announce a root cause before the fix is in and the number moves
metadata:
  type: feedback
---

In the commute transport test (2026-10-14) a string-replace edit meant to switch the PGD metric to the multi-batch
adversary rewrote the module docstring instead of the call, so a paid pod re-measured the old code. The agent then
told the user two different root causes (a grad-disabled decorator; adversary scope) before settling. The user said
the bugs "decrease my confidence in the robustness of your results, and the rigor of your experiments in general".

**Why:** a silent no-op edit costs a paid run and, worse, produces a confident wrong diagnosis; contradictory
explanations erode trust in every other number.

**How to apply:** after editing measurement code, re-read the function body (grep the call site) and have a CPU test
or a logged assertion prove the new path ran before launching anything paid. Report a cause only once the fix is in
the code and the number moved as the cause predicts; until then say "suspected". If an earlier explanation turns out
wrong, say so explicitly in chat and in the report. Rule also in `.claude/CLAUDE.md`. Related:
[[feedback-prereg-scoring-rigor]], [[feedback-define-terms-when-reporting]].
