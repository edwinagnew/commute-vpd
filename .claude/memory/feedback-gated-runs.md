---
name: feedback-gated-runs
description: "Runs over $15, over 3 h, or above the session cap are \"gated\" (.claude/CLAUDE.md, user 2026-10-28): pre-mortem, sourced settings, $0 rehearsal of every stage, early checks in code, plan+code reviews, user go-ahead"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 3f95fec9-dbd4-4457-aae1-901015e9f348
  modified: 2026-10-28T14:04:07.327Z
---

After E40's careless learning-rate mistake the user asked for "a special mode that allows more budget but has to pass significantly more stringent tests and checks so we dont get careless fuck ups like this again". They also said the first draft "over-indexes on step 1 learning rate issues. there are many things that can go wrong with big runs". The rule is in `.claude/CLAUDE.md` → "Gated runs".

**Why:** expensive Qwen runs cannot be iterated cheaply; every unchecked assumption (recipe, memory, data, saves, references) costs a paid rerun.

**How to apply:** give every gated plan a "Gates" section: a pre-mortem under each heading, a source for every setting, a $0 cluster rehearsal of every stage with the unknowns measured, early stop checks in code, a fresh-agent review of the plan and of the code, and the user's go-ahead after seeing the numbers. A restart with a larger budget is fine when the user offers it. Related: [[feedback-measure-step-size-before-scored-run]], [[feedback-compute-budget]].
