---
name: feedback-publish-cards-same-day
description: "Every finished research-log card is flushed and published the same day, never batched behind a running job (user, 2026-11-10)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 1138e853-a39e-4669-b08a-08a2adab9a5c
  modified: 2026-11-10T14:33:55.712Z
---

Every finished experiment card is flushed by the clerk and published the same day; never hold a publish to batch it with a job still running.

**Why:** 2026-11-10 I planned one publish after E143 landed; the user said "publish NOW, do not wait for E143", and made it a standing rule.

**How to apply:** when cards are committed, spawn the log clerk at once; the pending job gets its own flush later. Related: [[feedback-update-research-log-each-experiment]], [[reference-research-log-artifact]].
