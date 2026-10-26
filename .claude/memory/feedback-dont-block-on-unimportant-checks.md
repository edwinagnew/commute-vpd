---
name: feedback-dont-block-on-unimportant-checks
description: "A check that fails but doesn't bear on the experiment's key predictions shouldn't block the run; amend and go, ask if conflicted"
metadata:
  node_type: memory
  type: feedback
  originSessionId: b57cb0c4-c80c-4d62-b48f-b2f73d5a753f
  modified: 2026-10-21T13:58:40.351Z
---

Don't block an experiment because it fails a check that its main predictions don't depend on. Amend (dated amendment file filed before the rerun, as the repo does) and rerun; say what the failed check weakens. Don't abuse this: if the check might bear on the key result, or it feels like rationalising a miss, talk to the user first.

**Why:** 2026-10-21, E19's scored run stopped at the W₀ adversary floor (0.209 on the L40 vs a 0.15 line set from an H100's 0.075). Only P4/G1 used the floor; P1–P3 didn't. User: "dont block experiments if they fail on unimportant tests (but also dont abuse that exception - talk to me if you feel conflicted)".

**How to apply:** When a gate/control fails, first ask which filed predictions read it. If none of the core ones do, amend the gate and rerun without waiting; if any core one does, stop and check in. Also set card-dependent thresholds from the card's known values (the PGD floor is per-card: 4090 0.035, L40S 0.046, H100 0.075, A5000 0.122, L40 0.209). Never refile predictions themselves ([[feedback-never-refile-predictions]]).
