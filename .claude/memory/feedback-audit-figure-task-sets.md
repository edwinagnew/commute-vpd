---
name: feedback-audit-figure-task-sets
description: "drop the pair comparisons" applies to the tasks behind each figure, not just its wording; diff every figure's tasks x seeds x featurizations each full check
metadata:
  type: feedback
---
Fig 2 (retained share, E268→E277→E290→E298) read only E147's 5 key-grid tasks while Figs 3/7 and the tools table read all 16 known-answer tasks; noticed only 2026-11-27 (fixed by P1.66/E325). The user had asked many times to drop pair comparisons; it was applied to text and plots, never to the task selection inherited from the pair design (even known3.json is keyed by pair). E298's "make Fig 2 consistent" cell (11-24) defined consistency across featurizations only.

**Why:** user 2026-11-27: "reflect on how this is only getting noticed now, even though i must have said 'drop the pair comparisons' at least 10 times before".

**How to apply:** each full check, keep a table of every results figure's input tasks x seeds x featurizations x states and flag any figure whose set differs from the others; when a user rule changes framing, audit the data selections it implies, not only the prose. Related: [[feedback-same-runs-behind-compared-panels]], [[feedback-per-task-values-not-pair-gaps]], [[feedback-audit-drops-on-restructure]].
