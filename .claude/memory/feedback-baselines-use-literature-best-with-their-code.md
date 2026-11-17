---
name: feedback-baselines-use-literature-best-with-their-code
description: "A literature baseline is run with the best-performing published recipe and the authors' released code, not a cut-down reimplementation"
metadata:
  node_type: memory
  type: feedback
  originSessionId: df89eb39-f9c7-4cba-a63d-cb7601adcb7b
  modified: 2026-11-17T23:06:09.572Z
---

When adding a baseline from the literature, implement the best-performing technique as published and run it with the authors' released code (pinned commit), wrapping it for our activations; keep our own code to the glue.

**Why:** user 2026-11-17, after E198's crosscoder (own reimplementation, ~11x fewer tokens, a 0.9 bar the authors don't use for BatchTopK) stopped at its planted control: "you should implement the crosscoder experiment according to the best performing technique in the literature"; "if any existing papers publish their code, use that to reduce your mistakes". They approved ~$327 for the faithful run and also asked to confirm it is the strongest baseline and whether the token budget is needed at our scale.

**How to apply:** at /design for any baseline, find the strongest method that can give the reading (independent literature check), its released code and configs; list each departure with a reason; budget for the faithful recipe and ask if it is gated rather than shrinking it silently. See [[feedback-explicit-literature-departures]].
