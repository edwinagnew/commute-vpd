---
name: feedback-fable-first-fewer-questions
description: Route next-step/direction questions to Fable and act on its advice; ask the user only the most important questions
metadata:
  node_type: memory
  type: feedback
  originSessionId: bed69f2c-3f1e-41ad-84a0-6ab1759bfde0
  modified: 2026-10-30T14:48:04.049Z
---

"You're asking too many questions. Ask Fable for general advice and come to me only for the most important questions"
(user, 2026-10-30, after five AskUserQuestion dialogs in one afternoon on the re-gating line).

**Why:** the user had already adopted a programme; each next-step choice inside it came back as a dialog.
**How to apply:** for a choice inside an adopted plan (what to run next, ordering, filling a table, a blocked step's
workaround), spawn a Fable agent, take its recommendation, record it in the plan/decisions.md, and say what was done as
an FYI. Ask the user only for: spend above the cap or gated runs, a reversal of a filed prediction, a change of the
programme's goal, or another line's priorities (e.g. another lead's open ask). Related: [[feedback-dont-ask-for-cheap-recommended-checks]], [[feedback-decide-rule-fixes-with-clear-recommendation]].
