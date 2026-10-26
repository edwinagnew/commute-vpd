---
name: feedback-research-log-style
description: Research log boxes are headlines that point to the right figure; uninformative runs leave section 5; never quote an undefined number
metadata:
  node_type: memory
  type: feedback
  originSessionId: ec7102d2-9ad8-4634-b6f4-0a67011d1af0
  modified: 2026-10-21T11:33:28.132Z
---

The user's review of log v18 (2026-10-21):
- Q boxes and Results report headline results only and defer to experiments; a link to an experiment names which figure to look at, and that figure must show what the sentence claims.
- An experiment card's first sentence says why to care (the takeaway), and matches its first figure. Every card should have a plot.
- A number is either defined (X% *of what*) or left out: "be precise or stay high level", never feigned precision.
- Experiments sit in section 5 because they taught something, regardless of hardware (a no-GPU analysis can be a card, A1); uninformative runs (E13) and reviews go to section 11.
- Ongoing work gets a short card: what we measure, what went wrong, that it is being fixed.

**Why:** v18's boxes were too wordy and dense with numbers; the user couldn't tell takeaways from caveats.
**How to apply:** on every log edit, apply these; see [[feedback-update-research-log-each-experiment]], [[feedback-plots-over-tables]], [[feedback-writeups-explain-not-record]].

Also (user, 2026-10-21): "there should always be consensus across files about what state things are in." A status change goes into a dated decision report and then into every living file (README, decisions, TRIED, backlog, log, notebook header, memory) in the same pass.
