---
name: feedback-max-two-pods-per-runner
description: "Multi-pod runs get at most 2 pods per pod-runner; each finished pod is downloaded, sha-checked and terminated at once"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 55cd9995-fd95-4737-bb5b-1f97f764a9b8
  modified: 2026-11-21T03:17:34.480Z
---

A multi-pod run is split across pod-runners, at most 2 pods each. Each runner downloads, sha256-checks and terminates a pod the moment its item is done, before polling the others. If a pod has sat finished for more than ~5 min, spawn extra runners and split the pods.

**Why:** user 2026-11-20 19:15 (via whip-lead): one runner driving many pods on sae-lead's run left finished pods idle for a long time.

**How to apply:** every /pod-runner brief with more than 2 pods becomes several runners; check status() yourself when a runner reports finish times. Related: [[feedback-one-subagent-per-pod]], [[feedback-terminate-before-local-moves]].
