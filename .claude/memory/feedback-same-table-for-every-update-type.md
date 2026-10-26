---
name: feedback-same-table-for-every-update-type
description: User (2026-10-20) wants experiments less ad hoc - every update type measured by the same split table so techniques read side by side
metadata:
  type: feedback
---

On 2026-10-20 the user said: "I want our experiments to be a little less adhoc from now on so we get a consistent
picture of the techniques we are confirming."

**Why:** the outside share had been run on the fine-tune (E2, E8, E12) and the LoRA (E7, E8, E9) at different steps,
cards, tasks and under different headings, so no table put them side by side; the adjoint split and four-block
version had never been run on either.

**How to apply:** when a measurement is proposed for one update type (LoRA, fine-tune, random null), plan it for all of
them on the same task, steps, matrices and card, and say which of the standing techniques (least-squares split,
adjoint split, rescaling split, do-nothing, outside share) the run confirms. Before a new experiment, check
`backlog.md`'s "Split study" entry and `GOALS.md` Terms for what each technique has and hasn't been run on. See
[[feedback-design-against-goals]], [[project-lora-pullback-wanted]].
