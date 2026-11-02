---
name: feedback-fewer-questions-fable-for-small
description: Ask the user only the most important decisions; route smaller choices to a Fable agent and act on its advice
metadata:
  node_type: memory
  type: feedback
  originSessionId: 90e883cb-0787-4acc-805a-e16728a0d15e
  modified: 2026-10-30T14:26:30.843Z
---

Ask the user only the most important things; for smaller choices, ask Fable (`model: fable`) for advice and go with it, recording it as Fable's.

**Why:** 2026-10-30 the user answered a four-question AskUserQuestion (Q3 plan go-ahead, E45 Amendment 2 filing, drop E45 Job 2, scoring rule for behaviour kept) with "youre asking too many questions. please only ask me the most important things. ask fable for advice for smaller things".

**How to apply:** Before an ask or dialog, sort the items: plan-direction or money-above-cap choices go to the user (one question at most where possible); recipe details, gating readings of $0 jobs, rule wording, drop/keep of a sub-job go to one Fable agent, whose advice is acted on and noted in the plan/decisions.md as Fable's. Refines [[feedback-raise-decisions-often]] and [[feedback-dont-ask-for-cheap-recommended-checks]].
