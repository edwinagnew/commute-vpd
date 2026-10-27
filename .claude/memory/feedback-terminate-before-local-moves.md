---
name: feedback-terminate-before-local-moves
description: "Terminate a pod as soon as the result's sha256 matches; local file moves (into the results/large symlink) can block on a permission prompt for hours"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 1ed54cc2-156c-4342-9af4-d9a59e89c10d
  modified: 2026-10-27T10:22:00.163Z
---

Terminate the pod the moment the results JSON is home with a matching sha256; do every local move afterwards.
`results/large/` is a symlink to `~/commute-data/results-large/`, outside the project: download() into it is refused
and an `mv` into it waits at a permission prompt.

**Why:** 2026-10-26 the E36 A6 pod (H100, $2.89/h) finished in 2 min but billed ~14 h (~$40, the whole session
budget) while both the pod-runner and the lead sat at that prompt overnight. Recorded in CLAUDE.md's remote-kernels lessons.
**How to apply:** in every pod-runner brief, order "verify sha256 → terminate → move files"; when a runner goes
quiet with a live pod, check status() and terminate yourself once the result is home. Related: [[feedback-never-leave-a-job-unwatched]].
