---
name: project-success-criterion-reading-a
description: Since 2026-10-19 the commute success criterion is GOALS's weak version as written (decomposition passes VPD's checks on W+δW, Δ not absorbing δW); the route split d/r/i/e_gate/e_Δ is a mechanism reading, not the pass criterion
metadata:
  type: project
---

Decided 2026-10-19 (user: "lets stick with reading A so far. open to change at some point in future").
Reading A = GOALS weak version. Reading B = "the changed subcomponents carry the behaviour at the old gates" (direct
share d), which no map has passed and which is a mechanism finding under A. Recorded in
`reports/decisions/2026-10-19_decision_success-criterion.md` and a `decisions.md` row.

**Why:** from 2026-10-16 the scored numbers drifted to reading B while the weak version was checked on few states
(emoticon all metrics; one rung-0 task 1.09; one rung-2 task PGD 1.53, above the 1.5 line; four ladder tasks unmeasured;
VPD's sparsity loss never run as a metric).

**How to apply:** score new rounds on the task-local PGD ratio vs the W₀ floor, the share left in Δ, and the Pile
ratios first; report d, r as mechanism. Proposed but not made: mark GOALS's "Where an update goes" as mechanism
hypotheses (user-owned file). Related: [[feedback-check-metric-power-before-claiming]],
[[feedback-describe-methods-by-their-filed-definition]].
