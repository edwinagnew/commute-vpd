---
name: feedback-user-definitions-override-files
description: "The user's \"pull-back\"/\"LoRA pull-back\" means splitting a finished update over the subcomponents; the repo's file definitions (train the pieces) drifted from this and overrode the user for days"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 4aad360f-4019-4e9a-aa31-962e2c4a288f
  modified: 2026-10-20T13:16:57.978Z
---

Since 2026-10-14 the user has meant by "pull-back" (and "LoRA pull-back"): train a finished update (a LoRA's BA,
possibly to completion), split it into per-subcomponent updates, and measure how much lies outside the span of the
subcomponents. By "lift" they mean the least-squares way of choosing that split. They never asked for "train the
subcomponents on the loss", which is what GOALS Terms (2026-10-19), the theory note and five "LoRA pull-back" runs
(E7 pb_r, E13) built. Audit: `reports/reviews/2026-10-20_review_terminology.md`.

**Why:** "pull back" entered as the assistant's word on 2026-10-14; the file definitions then overrode the user's
repeated corrections (at least eight restatements over 2026-10-15 to 20), and each fresh session "corrected" the user
from the files. The user's confidence in the results was shaken by it.

**How to apply:** When the user's words and a file's definition of a term disagree, the user's words win and the
file is fixed first. Restate the user's request in plain operations (what is consumed: loss+data vs a finished δW;
what is done) and get that restatement confirmed before designing arms. Prefer the review's proposed names
(component training, least-squares split, adjoint split, outside share, LoRA split) over "pull-back"/"lift"
until the user settles the Terms. Report the outside share by the four blocks P_U δW P_V etc., not one span.
Related: [[feedback-describe-methods-by-their-filed-definition]], [[project-lora-pullback-wanted]].
