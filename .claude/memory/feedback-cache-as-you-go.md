---
name: feedback-cache-as-you-go
description: "Long runs must write every scored piece to disk as it finishes and be resumable; a small budget overrun is fine when the user says so, losing finished work is not"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 8a4a2c62-9d77-4228-b2d5-b557f8668031
  modified: 2026-10-20T15:55:47.130Z
---

On 2026-10-20, giving the go for the ~$10 Γ-refit run, the user said they "don't mind if you exceed $10 a little bit"
but to "cache as you go if possible so that if anything breaks we can resume later on".

**Why:** the 2026-10-19 pods lost unsaved states and a final JSON; a multi-hour run that restarts from zero costs more
than the overrun.

**How to apply:** a run script dumps its result JSON after every checkpoint and every stage, skips finished pieces on
re-run, and a re-run of an earlier stage must not overwrite later stages' results; the pod-runner downloads each piece
as it lands. A brief's budget line may say "a little over $10 is fine" only when the user said so in that session.
Related: [[feedback-compute-budget]], [[feedback-one-subagent-per-pod]].
