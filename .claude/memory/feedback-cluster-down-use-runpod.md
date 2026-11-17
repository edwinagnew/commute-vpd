---
name: feedback-cluster-down-use-runpod
description: When the MATS compute node is down, run the Python on a RunPod pod at once; never wait on the node
metadata:
  type: feedback
---
When cpu-run says the compute node is down, send the work (even summary/plot/lint) to a cheap RunPod pod through a pod-runner right away.

**Why:** 2026-11-16 the lead waited on the downed node over a weekend; user: "if the cluster is down, use runpod ... you need to move a bit faster now so please hurry up".

**How to apply:** label the pod with the run's E-code as `commute-E###-stage-what` (no space or colon; fleet `roles/shell-notes.md` → Pods); don't schedule a wake-up to wait for the node. Related: [[feedback-results-fast-runpod-default]].
