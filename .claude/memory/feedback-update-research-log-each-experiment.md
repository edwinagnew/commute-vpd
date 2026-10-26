---
name: feedback-update-research-log-each-experiment
description: "The research log (docs/research_log/parts/, artifact \"Update Decomposition\") must be updated and republished after every experiment and major decision, as part of the flow, not on request"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 6f8f2e79-c23b-4393-b1f5-8d2cf4341c62
  modified: 2026-10-20T13:46:29.230Z
---

On 2026-10-20 the user asked that updating the research log be part of the research flow: "this document should be
automatically updated after each experiment/major decision".

**Why:** the log is the colleague-facing explanation and the place the user checks that results fit together; a log
that lags the reports is worse than none.

**How to apply:** after an analysis report, a decision report, a new `TRIED.md` row or a dropped proposal, edit
the files under `docs/research_log/parts/` (experiment card, claims table, sub-question status, timeline, changelog row), run `build.py` and
republish to the URL in `docs/research_log/README.md`, in the same session. Run `build.py --check` against the live pages first; another
session may have published. No hook enforces this: the Stop hook was removed on 2026-10-20 because sessions kept
touching the log pointlessly after edits that changed nothing the log says. See
[[reference-research-log-artifact]].

**2026-10-21:** batched (user: "you can batch the log updates but make sure you dont miss things"). Each event appends a
line to `docs/research_log/PENDING.md`; flush before the session ends and at least daily, after the completeness
check in `docs/research_log/README.md` → "Batched updates".
