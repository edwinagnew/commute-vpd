---
name: feedback-dont-ask-for-cheap-recommended-checks
description: "A $0, ~15-min check that a report recommends and that only measures the unchanged model needs no For-you question; just run it"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 87710323-f323-4ec2-af39-fd66ff64ac7d
  modified: 2026-10-27T18:46:58.251Z
---

When a look-up or report recommends a cheap validation check ($0 cluster time, minutes, measuring the unchanged model, no design choice that a pending card could change), run it through /design → predictions → /pod-runner without asking the user first. Asking "should I run the fresh-numbers check?" got: "yes. not sure you needed to ask for this" (2026-10-27).

**Why:** the user values momentum; such checks fall in the approved classes and asking costs a round trip.
**How to apply:** reserve **For you:** for choices that change the plan (task switch, spend, reading of a result). Mention the cheap check as an FYI and start it. Related: [[feedback-dont-block-on-unimportant-checks]], [[feedback-raise-decisions-often]].
