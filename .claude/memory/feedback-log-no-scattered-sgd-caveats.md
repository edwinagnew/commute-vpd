---
name: feedback-log-no-scattered-sgd-caveats
description: Research log narrative must not be peppered with SGD/optimiser caveats; mention the SGD re-runs at most a couple of times; experiment cards self-contained
metadata:
  node_type: memory
  type: feedback
  originSessionId: 2024dc3c-9060-4b5c-811e-5fc305e38fa2
  modified: 2026-10-22T09:38:41.621Z
---

User, 2026-10-21 (to the log-rewrite agent): "low-level details like the choice of optimiser are not that interesting for most readers so please make sure the narrative is not suddenly peppered with annoying caveats about SGD. Instead, you should acknowledge this oversight in the 'What went wrong' section, add a small note in the E4 box that the relevant finding has been contradicted and then restructure the overall goals and flow in line with our new perspective." Clarified: "my earlier comment was about the narrative flow in the research log."

User, 2026-10-22: "the research log should acknowledge the SGD re-runs no more than a couple of times and mostly be self-contained when reading experiments boxes."

**Why:** caveats repeated across many boxes drown the takeaways; a reader of one experiment card should get its result without chasing E21 or the SGD story.
**How to apply:** when a finding rescopes earlier claims (as E21/E22–E25 do), fold the new result into each affected card's own verdict and Moved-on field in plain words, with no cross-reference chain; name the SGD re-run story at most twice in the whole log (e.g. once in Overview/claims, once in Mistakes or the timeline). See [[feedback-research-log-style]], [[feedback-update-research-log-each-experiment]].
