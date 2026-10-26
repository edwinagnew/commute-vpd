---
name: feedback-scope-a-grid-null
description: "A hyperparameter grid's failure is reported with its range, never as \"no value works\"; an unscorable prediction's failed precondition is not reused as a premise"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 07c22327-f9c1-4ccc-9061-161cc155f43c
  modified: 2026-10-21T14:04:05.779Z
---

E4 (2026-10-15) tried plain-SGD component training at rates 0.3–30, reported "no SGD learning rate works", and every
run for six days used Adam on that basis. The 2026-10-21 grid at 0.001–0.1 found SGD works at 0.003–0.1. The review
`reports/2026-10-21_review_sgd-finding-influence.md` traces what the wrong premise propped up (the "no frozen-Γ map
stays faithful" half of the main-line claim) and ranks the re-runs.

**Why:** a coarse grid whose best value fails at an edge, in the direction of that edge, has not tested the claim; and
nobody decided to rely on the null, it drifted into definitions (GOALS Terms "Adam, many steps") and a decision report.

**How to apply:** write a grid's null with its scope ("no rate in 0.3–30 by step 30"); extend the grid before writing
a null when the best value sits at an edge; when a prediction is unscorable because a precondition failed, say so
without generalising and put the fix in "Not run"; a later design that relies on such a null needs a `decisions.md`
row. Name the optimiser in every "held fixed / varied" line. See [[feedback-prereg-scoring-rigor]],
[[feedback-verify-before-announcing-a-cause]].
