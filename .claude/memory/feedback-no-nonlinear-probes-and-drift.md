---
name: feedback-no-nonlinear-probes-and-drift
description: "User dislikes non-linear (MLP) probes; built-control chain E148→E159 read as drift, each run reacting to the last puzzle"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 1195929b-5c0b-4175-a439-d34c95710bca
  modified: 2026-11-12T10:03:52.924Z
---

User 2026-11-12, after E159 (difficulty ladder): "i also dont like that youre using non-linear probes" and "where are we going here? feels like youre really losing track of what we want and/or pursuing things in the wrong way". They also could not see why the result was called a failure (the filed one-site "fall then level" was a floor-effect expectation, and the two-site share did fall then level).

**Why:** a chain of single runs (XOR2 control, sum parity, the H diagnosis, sum parity beside, the ladder), each answering the previous run's puzzle, drifted from part 1's question (a definition of reused against new that can be reported and defended) into describing the measure itself. An MLP probe can read almost anything, so it says little about what the model holds.

**How to apply:** use linear reads only, unless the user asks for more. Before proposing a follow-up run, say in one line how it moves the gate page forward, and prefer writing up what is known over another control. Don't call a result "against the filed prediction" unless the claim the prediction was made for actually failed. See [[project-working-definition-criteria]], [[feedback-resolve-dont-complicate]].
