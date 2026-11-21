---
name: feedback-be-pushy-on-pods
description: "When asked to check running pods, verify each pod's state and push the owning lead at once; one runner per ≤4 pods (user)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: e3a78ce2-b920-4f1f-860e-96de160ba84a
  modified: 2026-11-21T03:17:32.980Z
---

When the user asks me to check on running pods, I check the pods themselves, not just that "a runner is watching": for each pod, is it finished, idle, or downloading, and since when? Any finished or idle pod gets an urgent SendMessage to its owner lead straight away.

**Why:** 2026-11-20 ~19:15, the user said "you might need to be more pushy". sae-lead ran 8 E234 pods from one pod-runner subagent. Syncs and polls happened one at a time, so finished pods sat idle and results were badly delayed. I had reported "every pod has a watcher" as if that were enough.

**How to apply:** at every check, compare the pod list in fleet/check against how many runners each lead has. More than 4 pods per runner (the user told sae-lead ≤4) gets a poke: split the pods across more pod-runners and terminate each finished pod at once. "Has a watcher" is not "on track". Related: [[feedback-whip-lead-autonomy]], [[feedback-never-leave-a-job-unwatched]].
