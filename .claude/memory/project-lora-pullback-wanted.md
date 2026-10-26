---
name: project-lora-pullback-wanted
description: The user's LoRA pull-back = the LoRA split (train BA, split it over the subcomponents, measure the outside share); the adjoint split is their "pull-back", least squares their "lift"; built 2026-10-20 as experiments/lora_split
metadata:
  node_type: memory
  type: project
  originSessionId: 57e35906-4e49-4659-ac24-f800c1168fae
  modified: 2026-10-20T16:00:00.000Z
---

The user asked from 2026-10-14 to 2026-10-20 for the **LoRA pull-back**: "train a rank r lora matrix AB, possibly to
completion. Then split it into a sum of updates on subcomponents. Study how much lies outside the span of the
subcomponents" (2026-10-20). Five earlier runs (E7 pb_r, E13) built rank-r component training instead and were
deleted or re-labelled; see [[feedback-user-definitions-override-files]].

In the user's own words (2026-10-20): "pullback means stay within the components, lift means linear least squares".
So the **adjoint split** (dU = δW V, dV = δWᵀ U, Δ fixed) is their pull-back and the **least-squares split** their lift.
The adjoint split's weight change K(δW) is not δW, so it is applied with a per-matrix least-squares scale c on the data
(user-confirmed 2026-10-20).

**Status (2026-10-20):** built as `experiments/lora_split` (plan `reports/plans/2026-10-20_plan_lora-split.md`, design check
confirmed "as recommended"): no new LoRA training, E7 and E9's saved factors plus a re-run emoticon fine-tune at the same
nine steps; outside share as four input-weighted blocks that sum to one (read side split by E[a | Vᵀa]; GOALS Terms
updated), rescaling share, both splits' pieces saved, part 2 on Pile at steps 10/100/1000 on one card.

**Why:** agents kept dropping or reinterpreting the user's request; nothing flagged that it had gone.

**How to apply:** every LoRA design carries the split arm (both splits) or says plainly at the top that it has none and
asks. Never narrow the user's stated question without asking. See [[feedback-design-against-goals]],
[[feedback-same-table-for-every-update-type]].
