---
name: feedback-max-four-pods-per-runner
description: "At most 4 pods per pod-runner subagent; more left pods idle (E234, 8 pods from one runner)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 82634e48-ab9d-4032-a292-85571b74ffdf
  modified: 2026-11-21T03:13:47.106Z
---

Never give one pod-runner subagent more than 4 pods (user); whip-lead asks for 1-2 per runner with staggered syncs (2026-11-20 19:15), so use at most 2. For more parallel pods, spawn several runners (each ≤ 4 pods,
each with its own shards and labels).

**Why:** E234 (2026-11-20) ran 7 training pods plus an eval pod from one runner. Uploads ran one after another, finished
pods waited to be downloaded, and the eval pod was killed by the 30-min idle watchdog; the user saw several pods idle
and said "definitely no more than 4 pods from a single subagent" (user, 2026-11-20).

**How to apply:** when a brief splits a run over N pods, spawn ceil(N/4) pod-runners, or fewer pods. Applies with
[[feedback-one-subagent-per-pod]] (one runner per run still, with at most 4 pods each).
