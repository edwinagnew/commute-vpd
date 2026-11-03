# Plan: the least-squares read of the lowercase fine-tunes for the Q1 scorecard (E71)

*2026-11-03, commute-lead, on the user's task. Under the gate: cluster debug, ~12 min, $0. Not D1-D3: 67M work the user
asked for by name. Approved class 1 (GPU passes of the existing harness on saved states).*

**Restated in the Terms' words.** Read the least-squares split of the lowercase Adam and rank-8 LoRA fine-tunes (E49's
regenerated files) with E39's check, as a ratio to W0 in the same job, with do-nothing and the scaled adjoint split as
controls, and fill the scorecard's empty row. As a results-only row, read the expressible part alone on the same check.

**Design check.** *Goal:* `GOALS.md`'s question of whether an update can be written on the existing subcomponents; this
fills one cell of the slide that answers it on a broad task. *Nearest tried:* E16 read least squares on emoticon (6.96,
2.11; 36-38% in Delta); E39 read do-nothing and the adjoint split here but never least squares; E2 read the expressible
part's behaviour, never the check. *Patch or step:* a gap-fill for a slide. *Decisions:* none departed from. *What
changes what we do:* a control off by more than the band goes to the user before any paid rerun; otherwise the figure
updates. *Control first:* W0, gains, do-nothing, adjoint, each with a stop (`run.py`).

**Numbers and expected values:** `experiments/scorecard_ls/results/predictions.json`.

## Two points in the request that the code had to settle

1. **"The expressible part, Delta fixed" has two exact readings**, differing by the second-order term dU dV^T. Target
   W0 + δW − O(δW) with the least-squares pieces moves Delta by −dU dV^T (`expr`; its target is E2's "liftable" state).
   Target (U+dU)(V+dV)^T − UV^T + W0 leaves Delta exactly fixed (`expr_fixed`). Both are read; `expr`'s `delta_share`
   gives the term's size. The two are reported side by side, neither chosen.
2. **Least squares' "Behaviour kept" is 1.00 by construction**: its target weights are the fine-tune's, and that is what
   the measure reads. The expressible part's `s` is the informative one. "Left in Delta" is `harness.measure`'s unweighted
   energy share, the emoticon slide's measure, not the input-weighted outside share (kept in the results beside it).

## Card
E39's rows on the scorecard were read on an H100; this job is on an L40. "Close" means each control's check within 10%
of E39's (E49's `card_tol`, from four measured pairs) with the same verdict, and adjoint `s` within 0.02. The gap is
reported either way. A flipped verdict stops the job; a gap beyond the band without a flip is reported and the figure
waits for the user.
