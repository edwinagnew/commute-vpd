---
name: feedback-check-metric-power-before-claiming
description: "Before filing a prediction or calling a result \"faithful\"/\"works\", state what each metric is blind to and show a positive control moves it; inventory what was NOT run"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0277c70b-9c5b-479c-b4bd-ded756fa6198
  modified: 2026-10-15T09:22:42.551Z
---

On 2026-10-15 the user flagged "lots of red flags" and "so many sloppy mistakes" in commute rounds 1–2: every
faithfulness ratio was a mean over Pile rows holding ~3 trigger positions in 131k, so "the pull-back stays faithful"
could not see the trained behaviour; the pull-back had never been run on a LoRA although it was "the simplest thing to
try"; the pull-back's trained states were not saved; and nobody compared the update with Γ's values.

**Why:** a metric that cannot move when the claim is false makes a passed prediction vacuous, and an unrun arm quoted
as settled misleads the user about what is known. Both cost a round.

**How to apply:**
- In every plan, one "blind to" line per metric, and a positive control (a state that *should* fail) run before
  predictions are filed. Estimate the metric's dilution (effect positions / evaluated positions) explicitly.
- Measure a noise floor (seed-to-seed spread on the unchanged model) before setting any threshold.
- Every analysis report has a "Not run" list naming arms the question implies but the round did not do (e.g. LoRA
  through the pull-back), so gaps surface at the time rather than in the next session's reading.
- Save every trained state a report scores to `results/large/` before scoring it.
- Before asserting in chat that something was or was not measured, grep the run code; do not recall from reports.
- Never say "on every task" when coverage is partial: list the states a check was run on and their readings (2026-10-19: I said the pull-back passes faithfulness on every task; it had been checked on emoticon and two of six ladder tasks, one of which read PGD 1.53).
Related: [[feedback-verify-before-announcing-a-cause]], [[feedback-prereg-scoring-rigor]],
[[feedback-flag-qualitative-surprises-loudly]].
