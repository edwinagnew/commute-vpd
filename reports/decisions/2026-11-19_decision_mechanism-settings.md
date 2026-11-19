# Open: on which inputs do we compare a part's mechanism in the fine-tune against the base model?

*2026-11-19, definition-lead. Pending definition question, raised while planning the hand-built-edit run due 11-20.
`decisions.md` and `GOALS.md` are unchanged.*

## What was measured so far, and did it come out as expected

The working definition scores how much a fine-tune reuses the base model by comparing, part by part, the mechanism the
fine-tune uses with the mechanism the base model has, and reads one reuse score R from the parts the fine-tune needs.

- **On trained fine-tunes it did not come out as expected.** On 20 trained fine-tunes (analysis
  `reports/analyses/2026-11-18_analysis_r-bounds-e208.md`), comparing mechanisms by squared cosine over every position,
  any circuit of 128 or more neurons reads R of 0.99 or higher. The definition intends reuse and new to read apart;
  at the neuron level nothing separates.
- **Not yet measured:** whether the score separates reuse from new when the answer is known by construction. The pending
  run (E214, due 11-20) reads R on hand-built edits whose labels are known, at four levels: neurons and heads; heads and
  MLP layers; MLP and attention layers; blocks. Plan `reports/plans/2026-11-18_plan_p1-35-t2.md`, appendix
  `reports/plans/2026-11-18_plan_p1-35-t2_appendix.md` (section 4b and section 8, question 1). Nothing has read this yet.

## The question

When we compare a part's mechanism in the fine-tune against the base model, on which inputs? The hand-built edits change
a part's behaviour at only 1 of the 41 positions in each window (where the new cue is). So the choice decides how much
of the edit the comparison can see.

## The numbers it rests on

Each figure is the expected size of the gap between a reuse edit and a new edit, as estimated by the definition lead
when planning E214 (filed predictions: `experiments/kstar_levels/results/predictions.json`; plan appendix).

- Every position the part runs at in the data: the edit's change is diluted about 40-fold; reuse-against-new gaps
  of about 0.001 to 0.01.
- Only the positions where the behaviour is read: gaps about 40 times larger, but a change at another position in
  the window goes unseen.

## Options

- **A. Every position the part runs at in the data** (as the trained-fine-tune run did; it matches "parent settings
  that occur"). Tells us whether the score as already used separates known reuse from known new, at the cost of a small
  signal that may fall under the noise floor, as on the trained fine-tunes.
- **B. Only the positions where the behaviour is read.** Gaps about 40 times larger, so a clearer test. It would
  miss a mechanism that changed somewhere else in the window, and it departs from the comparison used so far.
- **C. Weight positions by how much each moves the behaviour.** Tells us the most about what matters, but it is a change
  to the definition beyond its current sentence, which only you make.

Whichever you choose, E214 reports the other of A and B as a side reading, so both numbers exist.

## Recommendation

**A, with B as the side reading.** It keeps the definition as written and the comparison with E208 and E213 direct, and
the side reading shows at once whether dilution is what hides the gap.

## Links

Plan `reports/plans/2026-11-18_plan_p1-35-t2.md`; appendix `reports/plans/2026-11-18_plan_p1-35-t2_appendix.md`;
predictions `experiments/kstar_levels/results/predictions.json`; analysis
`reports/analyses/2026-11-18_analysis_r-bounds-e208.md`.
