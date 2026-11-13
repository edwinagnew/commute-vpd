# Decision: take the "simplest intervention" wording of reuse forward?

*2026-11-13, definition-lead, asked for by the user. Log card `decision-fourth-wording`. It replaces the card
`decision-third-wording-gate` (`reports/decisions/2026-11-13_decision_third-wording-gate.md`). That card recommended
adopting the third wording without saying that it is not total, so its recommendation is withdrawn. Full wording:
`~/commute-data/scratch/definition-session/2026-11-13_reuse-as-simplest-intervention_v4.md` and the log's glossary
entry `def-reuse-simplest`. Nothing in `decisions.md` changes until you choose. Tentative: you have not yet seen the
wording in full.*

## The question

A new wording keeps the third wording's one line and adds a grade:

> **F reuses M to the extent that F is M under a simple intervention on M's variables. Reuse is graded by the simplest
> such intervention: the shorter it is to describe, given M, the more F reuses M.**

Should this be the wording we test next?

## What it fixes

- **Total.** Every fine-tune gets a grade. At worst the intervention rewrites the output from scratch, so there is no
  "no reading". This was the failure of the third wording and of Jain et al.
- **No unlimited interventions.** Any intervention is allowed, but each is charged for its complexity in bits. Doing
  "anything" therefore costs as much as describing the change from scratch, which reads as "new".
- **General.** The definition does not mention linear reads, affine maps, a hand-named part or a found causal model.
  Those are measurement choices, and they live in a separate section.

## How complexity is compared

The cost of an intervention is the bits needed to:
- name which of the original's variables it touches (cheap when the original already computes them);
- write down its numbers at a fixed precision.

The intervention must reproduce the fine-tune within a fixed KL tolerance. KL measures fidelity, not complexity.
Reuse share = 1 − (cost using the original's insides) / (cost of an output-only rewrite).

The three grades come out of the bit count rather than being set by hand:
- **re-setting:** a constant, gain or shift, costing a few bits;
- **new trigger:** reading other variables, costing a few more;
- **new:** costing about as much as starting from scratch.

## Expected readings (by argument, not measured)

| Case | Expected reading |
|---|---|
| Do-nothing | cost 0, share 1 |
| Built re-setting | share near 1 |
| Push gated by parity | near 1, new trigger |
| Random-bit floor | near 0, new |
| Memorised fact | near 0, unless the original already has a variable that picks out the trigger, in which case new trigger |

## Limits

- **Relative to a description language.** In practice that means the variable list, the precision, the KL tolerance
  and the inputs. This is stated, not removed.
- **Every number is a bound.** The true minimum cannot be computed, so a measured "new" means "nothing simpler was
  found". A better search can only move a reading toward reuse. The second wording's searches stalled between 0.47 and
  0.84 on edits that were exactly linear, so search quality matters.
- **The dictionary problem.** On a small input set, a memorised fact is one linear direction of token identity. The
  measurement must charge for writing that direction down, or the fact reads as cheap.
- **A possible circularity.** If the variable list is the decomposition's components, the measurement resembles
  re-gating. The definition does not, because the list is only one way to search.

## Options

- **(a) Test this wording next. Recommended. About a day, on the cluster or a pod.**
  - Process: I write a short plan with the `/design` check, and it runs once you open a cell for it.
  - What it measures: the cost-against-KL curve on the built edits (the controls run first), the fact, its twin and
    the old matched pairs.
  - What it tells you: whether the built edits grade as built, and where the real fine-tunes fall on the scale.
  - What happens next: you adopt the wording, or revise it, before part 2 starts.
- **(b) Refine the wording first, with no run.** A fresh reviewer and you go over the precise form, then (a). This
  costs a day of waiting, but catches flaws before any code is written.
- **(c) Keep the second wording.** No spend. Part 1 then never says "built" on small input sets.

## Recommendation: (a)

The wording is total and general, and its readings on the built edits can be checked against known answers in one
run. A review can run alongside the plan rather than before it. Choices only you make if it goes ahead:
- the variable list (neurons, components, or principal directions on ordinary inputs);
- the precision in bits;
- the KL tolerance;
- whether to report the share or the whole curve.

Names in the reports: the third wording's tests are cells P1.18 and P1.19 and experiments E173, E176 and E177; the
second wording's adoption is the `decisions.md` row of 2026-11-10.
