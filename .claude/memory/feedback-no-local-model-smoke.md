---
name: feedback-no-local-model-smoke
description: Subagents must not load the 67M model/decomposition locally; concurrent CPU smokes OOM-killed the laptop (2026-10-21)
metadata:
  type: feedback
---
Writer subagents never run anything locally that loads the model or the VPD decomposition (no CPU smoke); only lint, light unit tests run one at a time, and imports. The smoke runs on the cluster GPU via a pod-runner.

**Why:** 2026-10-21, three parallel writer agents ran CPU smokes; the 14 GB laptop hit the kernel OOM killer and Claude Code crashed, killing every agent.

**How to apply:** put this rule in every code-writing brief; the GPU smoke is the pod-runner's first step. Related: [[feedback-one-subagent-per-pod]], [[feedback-big-runs-on-gpu]].
