---
name: project-elicitation-granularity-limit
description: "VPD's \"elicited\" means \"uses W₀'s existing rank-one pieces\", not \"uses an existing circuit\"; state this limit in every re-gating claim"
metadata:
  node_type: memory
  type: project
  originSessionId: 14903db7-9605-4ef7-85fe-37f7f5b58b6b
  modified: 2026-10-29T13:45:20.028Z
---

A new mechanism assembled from subcomponents W₀ already has, under new gating, reads as "elicited" (re-gating) to VPD,
because VPD's unit is the rank-one subcomponent, not the circuit. It is a limit of the method, not a bug to fix; the
only check that narrows it is a Γ from a second decomposition seed. Recorded in decisions.md (2026-10-29).

**Why:** the user called this "very important" and said they could easily forget it (2026-10-29, on Fable's
re-gating elicitation plan, reports/plans/2026-10-29_plan_restore-anchors.md).

**How to apply:** every re-gating write-up, log claim and plan that reads "elicits vs new" states this limit in plain
words; remind the user of it when presenting re-gating results. Related: [[project-two-parallel-lines]].
