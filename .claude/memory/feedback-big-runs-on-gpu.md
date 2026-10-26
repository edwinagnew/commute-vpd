---
name: feedback-big-runs-on-gpu
description: "Anything that trains more than one benchmark network goes to the MATS cluster or RunPod, never the local CPU — theory check scripts included"
metadata:
  node_type: memory
  type: feedback
  originSessionId: c2f8cc57-50e7-46ef-84bc-116f06dbf92d
  modified: 2026-10-12T11:50:00.896Z
---

Run training remotely (user, 2026-10-12, repeated several times: "dont run heavy cpu tests. go to cluster or runpod! (why do i have to say this every time?)"). Local CPU is only for: one benchmark net of under a minute, closed-form / no-training checks, and analysis of saved results.

**Why:** the user has had to repeat this more than once; an earlier version of this note exempted "theory checks", and a check script that trains a grid of nets was then run locally. Long local runs block the session and the machine.

**How to apply:** if a script trains networks (experiment, theory check, sweep, "quick" slice of several nets), benchmark one net locally, then submit it to the MATS cluster first ([[project-mats-cluster]]; RunPod via remote-kernels if the queue blocks, [[project-runpod-remote-kernels]], [[feedback-pod-names]]). Confirm first if the estimate exceeds $10 ([[feedback-compute-budget]]).

**Ten-minute rule (user, 2026-10-12):** estimate wall time from the one-net benchmark *before* submitting. If the job is expected to take over 10 minutes, it goes to a GPU (RunPod), not a cluster CPU job. State the estimate when submitting. Batch many tiny nets into one GPU process (as `train_torch` does) so the GPU actually helps.
