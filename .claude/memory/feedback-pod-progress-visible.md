---
name: feedback-pod-progress-visible
description: "On RunPod, stage progress must be visible (container stdout) and waits must read the log every 1-2 min with a stall rule; never blind sleep loops, never hour-long blocking executes; set-up gets a deadline"
metadata:
  node_type: memory
  type: feedback
  originSessionId: c0a515b9-ffbf-4017-ba07-96999c48b12f
  modified: 2026-10-19T14:31:46.060Z
---

During the first transport-test pod runs (2026-10-14) the user intervened twice: they wanted to see progress in the
RunPod web console, not only inside kernel cells, and they read an hour-long blocking `execute()` as a freeze.
On 2026-10-19 (reading-A coverage pod) a pod-runner agent waited in blind `sleep(10)` loops of 14 minutes with no
log reads; a failed `uv sync` (dropped GitHub clone) was noticed 11 min late, a retry was launched twice into the
same venv, and the user asked "the subagent was waiting for ages while nothing was happening. how can we avoid that?"

**Why:** the user watches pods from the RunPod console and the agent transcript; a silent container, a blocked tool
call and a sleep loop all look identical to a hung run, and every idle minute bills.
**How to apply:** the stage runner on a pod tees each stage's log into the container's stdout (`/proc/1/fd/1`) from
the start; long stages run with `execute(background=true)` or a subprocess; every wait reads the log every 60-90 s,
prints the new lines, exits on failure signatures (`error:`, `Traceback`) as well as on process exit, and treats a
log (or cache size) that has not grown for 3 min as a stall to report, not to wait through. Briefs carry a set-up
deadline (~12 min: report and stop). Kill by PID, not `pkill -f` from a subprocess shell (it matches itself). Prefer a
prebuilt network volume (venv, artifacts, caches on `/persist`) over installing on every pod. See
[[feedback-one-subagent-per-pod]].
