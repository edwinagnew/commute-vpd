---
name: feedback-scored-run-is-its-own-smoke
description: Since 2026-10-21 (user) no routine smoke runs; the scored run with its W₀ control first is the smoke; separate smoke only for expensive failures
metadata:
  type: feedback
---
The scored run is its own smoke. Separate smoke only for a paid run over ~30 min or an expensive late stage; new experiments need no `--smoke` path. Rule lives in `.claude/CLAUDE.md` → Writing code.

**Why:** user, 2026-10-21: smokes "very regularly cause issues and almost never do anything useful" — a second queue wait each, smoke-size config code, uninformative "failed G1 as expected" logs, and local CPU smokes OOM-killed the session.

**How to apply:** pod-runner briefs say "scored" by default; on a first-run code error the runner reports the traceback. Related: [[feedback-no-local-model-smoke]], [[feedback-one-subagent-per-pod]].
