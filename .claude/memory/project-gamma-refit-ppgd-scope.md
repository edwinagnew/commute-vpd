---
name: project-gamma-refit-ppgd-scope
description: "Γ refit (experiments/gamma_refit, 2026-10-20) keeps one persistent-PGD state per micro-batch slot so the released per_batch_per_position adversary scope survives gradient accumulation"
metadata:
  node_type: memory
  type: project
  originSessionId: 8a4a2c62-9d77-4228-b2d5-b557f8668031
  modified: 2026-10-20T15:47:03.154Z
---

The Γ refit (`src/commute/gamma.py`, built 2026-10-20 from `reports/2026-10-20_plan_gamma-refit.md`) trains Γ alone on
the released loss stack minus faithfulness. The released config's persistent-PGD adversary has scope
`per_batch_per_position` (one source per row-and-position slot of the 64-row batch, one step per optimizer step). With
micro-batches of 8, a single state stepped per micro-batch would give the adversary 8× the updates per Γ step; the code
holds `batch // micro` states, one per slot, each stepped once per step (Codex adversarial review, 2026-10-20).

**Why:** the test is "like for like with how Γ was fitted at W₀"; a stronger adversary per step changes the objective.

**How to apply:** when changing `batch` or `micro` in `experiments/gamma_refit/run.py`, remember the sources cost
~5 GB plus Adam moments (~15 GB) at batch 64 regardless of micro; an H100 is needed. See [[project-mats-cluster]].
