---
name: feedback-same-runs-behind-compared-panels
description: "Panels compared side by side must come from the exact same experiment (same states, seeds, randoms, conventions); unequal point counts are a red flag"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 4db6611b-3907-4c58-b456-e0e6b8a1cfcb
  modified: 2026-11-24T18:56:06.511Z
---

Any plot comparing featurizations or readings must show the exact same experiment on each: the same states, seeds,
matched randoms and reading conventions, so the point counts match (user, 2026-11-24, on Fig 2: VPD had seed 0 only,
top neurons 3 seeds, SAE the scaled states).

**Why:** "differing numbers of points is a glaring sign of non-standardised tests."

**How to apply:** before a comparison figure goes in the paper, count the points per panel and check the source runs.
If they differ, get one standardised run via whip-lead (see [[feedback-defer-to-whip-lead-on-priority]]) rather than
explaining the gap in the caption. Related: [[feedback-same-table-for-every-update-type]].
