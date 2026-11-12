---
name: feedback-short-go-ahead-questions
description: "Gate-6 go-ahead dialogs must be one or two lines (cost, wall time); the user found a full gates summary \"too long\" and said \"just run\""
metadata:
  node_type: memory
  type: feedback
  originSessionId: 3df1a2c8-4061-48bf-b912-63f17094fece
  modified: 2026-11-12T11:47:59.170Z
---

A go-ahead question for a gated run is short: what runs, cost, wall time, one line. Details stay in the gates file.

**Why:** 2026-11-12, E167 go-ahead dialog listed cost breakdown, rehearsal numbers and stop rules; user: "too long of a question. i think just run".

**How to apply:** AskUserQuestion text ≤ ~2 lines with a link to the gates file; put the breakdown in the plan, not the dialog. Related: [[feedback-short-chat-replies]], [[feedback-gated-runs]].
