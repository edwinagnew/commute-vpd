# Adopt the second wording of reuse as the working definition, with its labels read within matched pairs?

*2026-11-10, definition-lead, gate of part 1 of the paper plan, cell P1.1. This is a gate decision: it is put without a
recommendation. Page: `reports/analyses/2026-11-10_analysis_gate-page-working-definition.md`. Figure:
`experiments/old_task_labels/figures/gate_within_pair.png`.*

**Decided 2026-11-10 (user, last row of `decisions.md`).** The second wording is adopted tentatively as the working
definition. In the user's words: "yes you can tentatively adopt the 2nd wording as the working definition of re-use. if
it cant detect built, then it will have to be revised however." The cut-offs are rounded from 0.794 to 0.8 and from 0.476 to 0.5. At the
rounded cut-offs one label moved: the 300-step copies of Always on now read no map found, where they read rewired. No
old 67M task moved. The built control runs as a new cell. The numbers below are at the rounded cut-offs.

**The wording, in one paragraph.** A fine-tune reuses the original when some small variable of the original carries the
whole change in behaviour, and the fine-tune sets it by a fixed affine map of the variable's own value: turned up, on by
default, shifted. That is "re-weighted". If the variable is set from everything the original had computed, the change is
"rewired"; if it needs something the original had not computed, it is "built". Two shares measure this: the
re-weighting share, how much of the change one four-direction variable of the original explains, and the rewiring
share, the same from everything the original computed. Each is 0 for nothing explained and 1 for all of it.

**What was done.** The wording was read, with a blind search, on four hand-made changes to the 67M model, on ordinary
fine-tunes trained to copy them, and on the seventeen old 67M task fine-tunes. Each had a label fixed by how it was built.
Nothing new ran for the gate.

**What the numbers say.**
- On the hand-made changes the search found the known answers: 0.922, 0.961 and 0.995 against a bar of 0.9.
- The table copies read rewired on all three seeds, their built label. At the rounded cut-offs, the 300-step copies of
  the re-weighted change read no map found, with two of three rewiring shares under 0.5; the 3000-step copies read
  rewired. Their re-weighting share stayed under the 0.8 cut-off, at most 0.78 after ten times longer training.
- In all five matched pairs of old tasks, every seed built to reuse has a higher re-weighting share than every seed of
  its new twin, by at least 0.17. Under the 0.8 cut-off, only the real case label reads re-weighted; the other reuse
  sides read rewired, like their twins, except lowercase, which reads no map found. No old task changed label when
  the cut-offs were rounded.
- "Built" is never read: a low rewiring share only says the search found no map. The control that could change that is
  designed, not run.
- The two shares read the outputs only. The one number that looks inside, the reversion, stayed from 0.24 to 0.26 on the
  copies, against 0.938 on the hand-made change. Nothing independent yet says which is right for the copies.

**The question.** Adopt the second wording as the working definition, with labels reported within matched pairs, or not?

- **Option a: adopt as is.** A dated `decisions.md` row makes it the working definition; results are reported as the
  re-weighting share of the reuse side against its new twin, and absolute labels are not given. It tells us the comparison
  of re-gating, the neuron reading and simple baselines on the 67M, and the Qwen repeat, are scored against a
  within-pair share. Next: part 2 starts on that basis.
- **Option b: adopt, with the open cut-off as a follow-up cell.** As option a, plus a new cell to find whether one
  absolute cut-off can label ordinary fine-tunes, filed before it is checked on fine-tunes it was not set on, and the
  built control run. It tells us whether absolute labels and a "built" reading are possible. Next: part 2 starts, and the
  follow-up cell runs beside it.
- **Option c: reject.** The wording is not adopted. Parts 2 and 3 have no definition to score against: the comparison on
  the 67M and the Qwen repeat wait while a new wording is written and tested, or are reported against construction labels
  only. It tells us the share's within-pair ordering is not enough for a definition. Next: back to the wording and the
  built control, part 1's earlier rungs.

**Files.** Page `reports/analyses/2026-11-10_analysis_gate-page-working-definition.md`; old-task results `experiments/old_task_labels/results/labels.json`; wording
`~/commute-data/scratch/definition-session/2026-11-09_reuse-at-a-variable_v2.md`; built control
`reports/plans/2026-11-10_plan_built-control-design.md`; today's within-pair decision
`reports/decisions/2026-11-10_decision_old-task-cutoff.md`.
