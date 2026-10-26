---
name: feedback-results-fast-runpod-default
description: commute GPU routing since 2026-10-20 — cluster debug only for runs ≤ 30 min; longer runs straight to RunPod on a 4090 or faster (H100 > ~1 h)
metadata:
  type: feedback
---

The user wants results as quickly as possible. Runs of ≤ 30 min go to the cluster's `debug` QoS, and to RunPod if
they haven't started within 10 min. Longer runs go straight to RunPod on a 4090 or faster card (an H100 for runs
over ~1 h). Never pick an A5000/3090-class card to save money. The $10/session budget still applies.

**Why:** pod-runners were too wary of RunPod and chose excessively small GPUs. The trade-off is a ~5-min pod set-up
against the cluster's queue: `debug` starts in 3–6 min, `normal` is days.
**How to apply:** write it into pod-runner briefs (Platform line), and don't revert to cluster-first from the shared
`../CLAUDE.md`. Rules: `.claude/agents/pod-runner.md`, `decisions.md`. Related: [[project-mats-cluster]],
[[feedback-one-subagent-per-pod]], [[feedback-compute-budget]].
