---
name: feedback-decide-rule-fixes-with-clear-recommendation
description: "A filed control/rule that is plainly mis-specified, with one clearly recommended fix, is fixed by the lead (dated amendment) without asking"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 8d1e891c-cfa9-4ba7-b9b7-8746dc0e1bed
  modified: 2026-10-29T10:01:59.531Z
---

When a filed control or scoring rule turns out mis-specified (cannot pass by construction, or confounded) and the lead
has one clearly recommended fix (here also backed by Fable), file the fix as a dated amendment before the scored stage
and rerun, then tell the user as an FYI. Don't file an ask.

**Why:** 2026-10-29, E46: the user answered asks on a two-sided null that could not pass and on P0.4's confounded copy
writes with "A on both, rerun 0a. not really worth asking me".

**How to apply:** still ask when the options would change what a result means for GOALS.md, or when the
recommendation is not clear-cut. Never edit a rule after its scored run has read it ([[feedback-never-refile-predictions]]).
Related: [[feedback-dont-block-on-unimportant-checks]], [[feedback-dont-ask-for-cheap-recommended-checks]].
