---
name: feedback-compute-budget
description: Session compute budget is REMOTE_KERNELS_BUDGET ($40 since 2026-10-21); no per-run sign-off below it; confirm anything above
metadata:
  type: feedback
---

The session budget is `REMOTE_KERNELS_BUDGET` in `.claude/settings.json`: $40, confirmed by the user on 2026-10-21
("yes $40 is the new number"), up from $10. Below it, runs need no per-run sign-off; above it, estimate and ask.

**Why:** the user wants spend bounded but no longer wants a check-in per run (2026-10-21, moving to approved classes
of work, see [[feedback-design-against-goals]]). The plugin terminates machines when the budget is hit, so it is a hard
backstop, not a target.
**How to apply:** benchmark a slice first, extrapolate, and plan to stay ~15% under what is left of the $40. Pod-runner
briefs set their cap from the remaining budget. Never raise the variable without being told to.
