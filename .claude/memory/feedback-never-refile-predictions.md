---
name: feedback-never-refile-predictions
description: "Never offer to refile or flip a filed prediction, even before the run; a wrong prediction is explained, not edited"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 01d9293b-dfb6-4f6a-bd38-da5732291b3d
  modified: 2026-10-19T13:46:42.332Z
---

Never offer to "refile" a prediction as pass or fail, even when nothing has run yet. Once filed, the prediction is
the prediction: it passes or it doesn't, and a wrong one gets its cause worked out in the analysis report.

**Why:** the user (2026-10-19): "i will almost never want a prediction 'refiled as pass' - either it passed or it
didnt! If your prediction was bad, you must work out why rather than trying to brush over it." Offering to flip a
prediction treats pre-registration as paperwork and hides what the model of the system got wrong.

**How to apply:** file what you actually expect, say so once, and stop. After the run, score as filed; for a miss,
the report names candidate causes and what would separate them (see [[feedback-prereg-scoring-rigor]],
[[feedback-flag-qualitative-surprises-loudly]]). Changed rules go in a dated amendment file only when a measurement
was impossible as filed, never to change a verdict.
