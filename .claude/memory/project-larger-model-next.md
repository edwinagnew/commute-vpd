---
name: project-larger-model-next
description: "The 67M Pile model is a temporary testbed; the project will move to a larger model soon, and the project is called \"Update Decomposition\" in write-ups, not \"Commute\""
metadata:
  node_type: memory
  type: project
  originSessionId: 6f8f2e79-c23b-4393-b1f5-8d2cf4341c62
  modified: 2026-10-20T11:41:09.675Z
---

The user said on 2026-10-20 that the 67M model is temporary and the project will soon move to a larger model, and that
outward-facing write-ups should be titled "Update Decomposition" rather than "Commute" (the repo keeps its name).

**Why:** the 67M model was chosen only because its VPD decomposition is released and runs take minutes; the
question is about real post-training on real models.

**How to apply:** do not describe the 67M model as "the model" in titles, subtitles or summaries; call it the current
testbed. Sub-question lists and plans should carry the move to a larger model as a phase. Harnesses written against the
released decomposition (floors, saturated matrices, the shared Γ) will need re-validating on the new one. See
[[reference-research-log-artifact]].
