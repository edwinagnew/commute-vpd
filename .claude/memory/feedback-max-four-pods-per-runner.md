---
name: feedback-max-four-pods-per-runner
description: "At most 4 pods per pod-runner subagent; more left pods idle (E234, 8 pods from one runner)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 82634e48-ab9d-4032-a292-85571b74ffdf
  modified: 2026-11-21T03:13:47.106Z
---

Never give one pod-runner subagent more than 4 pods (user); stagger the uploads. whip-lead briefly asked for 2, then deferred to the user's 4 (2026-11-20). For more parallel pods, spawn several runners (each ≤ 4 pods,
each with its own shards and labels).

**Why:** E234 (2026-11-20) ran 7 training pods plus an eval pod from one runner. Uploads ran one after another, finished
pods waited to be downloaded, and the eval pod was killed by the 30-min idle watchdog; the user saw several pods idle
and said "definitely no more than 4 pods from a single subagent" (user, 2026-11-20).

**How to apply:** when a brief splits a run over N pods, spawn ceil(N/4) pod-runners, or fewer pods, and stagger their uploads. Applies with
[[feedback-one-subagent-per-pod]] (several runners per run are fine, at most 4 pods each).
