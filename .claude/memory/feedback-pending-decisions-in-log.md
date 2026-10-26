---
name: feedback-pending-decisions-in-log
description: Choices waiting on the user go as cards in the research log's "Pending decisions" section (section 3, `parts/sections/decisions.html`), coherent on their own; removed once decided (2026-10-22)
metadata:
  type: feedback
---

Every decision waiting on the user is a card in `docs/research_log/parts/sections/decisions.html`: what the choice
is about in plain words, what the evidence says, the options and what each changes, why it matters, then links to
reports/notebooks/cards. Remove the card once decided (decision → `decisions.md` + changelog).

**Why:** 2026-10-22 a pending item was handed to the user as "whether H′ is restated with the scope the kill tests
give it: holding on tasks the model already does, failing on the tasks it lacks, in line with E12's finding…"; the user:
"that makes no sense and is not a coherent question. i already asked that such questions be phrased more clearly".
Pointers are fine, but the card must make sense without them.

**How to apply:** never say "the user's call" in a report or chat without a matching Pending-decisions card; phrase the card's
title as a plain question a newcomer could answer. Recipe in `docs/research_log/README.md` → "Pending decisions".
Extends [[feedback-plain-approval-requests]]; see [[feedback-user-reads-log-and-chat-only]].
