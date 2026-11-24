---
name: feedback-literature-methods-spec-first
description: "any literature baseline needs a quoted method spec from the paper, plus a departures table, checked by redteam and a GPT reviewer before code; cite the method by name"
metadata:
  node_type: memory
  type: feedback
  originSessionId: db10179f-7398-49f1-b4a8-db985dc150db
  modified: 2026-11-23T23:37:16.511Z
---

On 2026-11-23 the user found three problems with our baselines:
- Our "ADL" baseline (E264-267) scored a quantity, D, that may not be in the paper. Its citation pointed to the crosscoder paper.
- "Neuron-level activation patching" was our own composite of two papers.
- The steering share is Venhoff 2025's method but was not cited as such.

The user's words: "how in the world does this happen? ... i told you to be extremely rigourous when implementing other paper's techniques"; "im really concerned now about the legitimacy of our baseline experiments".

**Why:** a reviewer will judge the paper on whether its baselines are the real published methods. A homemade version labelled with a paper's name is worse than no baseline at all.

**How to apply:**
- Before any code for a method named after a paper, the lead writes a spec that quotes the paper for each step.
- The spec includes a departures table (ours/theirs) and uses the authors' pinned code where it exists.
- redteam-lead and an independent GPT reviewer check the spec before any run.
- If our quantity is an adaptation, name it as ours, with "after X", never as X.
- When one of our readings matches a published method, cite it by name.
- Extends [[feedback-baselines-use-literature-best-with-their-code]] and [[feedback-explicit-literature-departures]].
