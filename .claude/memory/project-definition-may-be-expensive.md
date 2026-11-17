---
name: project-definition-may-be-expensive
description: "The reuse definition may be slow/exhaustive to evaluate; the paper's methods are its cheap approximations, so never cap a definition's search for cost"
metadata:
  node_type: memory
  type: project
  originSessionId: 92542d5a-2687-4cdc-818a-cfe8e67f83d6
  modified: 2026-11-16T21:55:56.885Z
---

The working definition of reuse (any wording) is allowed to be expensive: a slow, exhaustive search is fine. The point of the methods compared against it (re-gating, the neuron reading, baselines; part 2 of the paper) is that they efficiently approximate a potentially expensive definition (user, 2026-11-16, asking if I remembered; also 2026-11-13 on the fourth wording: "be thorough, exhaustive and precise").

**Why:** a search capped for speed (E184's 16-name cap, 4 candidates) made built items read too high, so the definition's reading was confounded with the search's limits.

**How to apply:** when designing a definition's measurement, size the search for exactness (exhaustive candidates, uncapped lists, long refits, H100s/hours are fine), not for runtime; "too slow" is never a reason to cut a definition's search. Speed limits belong to the approximating methods. Related: [[project-working-definition-criteria]], [[feedback-definition-must-be-total]].
