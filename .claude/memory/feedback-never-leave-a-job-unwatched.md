---
name: feedback-never-leave-a-job-unwatched
description: "A submitted cluster/pod job always has a live watcher; if a runner hands back with a job still queued or running, spawn a fresh watcher at once instead of offering the user a cancel-or-wait choice"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 2024dc3c-9060-4b5c-811e-5fc305e38fa2
  modified: 2026-10-22T16:44:04.617Z
---

2026-10-22: E31's runner submitted a shard rerun (job 29596), was then blocked on a read-only squeue, and handed back with the job unwatched. The lead relayed it as "cancel it or let it finish, your call". User: "get a fresh subagent to monitor the remaining cluster job. im confused why you stopped without it in the first place".

**Why:** an unwatched job's results never come home on their own, and the user expects the lead to finish the run, not to hand back the loose end. A $0, unscored job is still part of the deliverable.
**How to apply:** whenever a runner's hand-back says a job is still queued or running (or a pod is still up), spawn a fresh `pod-runner` watcher immediately with the job ID, outputs and the post-run steps (sync down, merge, plot, report's "Not run" line, PENDING line). Mention the watcher in the reply, never a choice. See [[feedback-one-subagent-per-pod]], [[feedback-pod-progress-visible]].
