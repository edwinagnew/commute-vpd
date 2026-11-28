# Does the adjoint split, with or without its scale, separate the built edits more cleanly than least squares?

We read the gain share on seven hand-built edits, splitting each weight change by least squares and by the adjoint
split, without and then with its scale. Without its scale the rescaled edits fell to a fifth and rebuilt fell to the
new edits' level; with it, the edits kept the least-squares order at lower values. So the adjoint split as `GOALS.md`
defines it orders the built edits as least squares does, but carries only half of rebuilt's change.

## The answer

Splitting each built edit's weight change by the unscaled adjoint split instead of by least squares makes every edit
look less like rescaling, not more. The two gain edits fell from 0.761 to 0.207. The rebuilt edit fell from 0.623 to
0.012, next to the new edits at 0.013. The likely cause, not checked here, is that the unscaled split writes a change
many times the real one onto the subcomponents and cancels it in the remainder. Correcting that is the scale's job.

With its scale, the second run, the adjoint split puts the edits back in the order least squares gives. The gain edits
read 0.71 and the rebuilt edit 0.45, both a little below least squares. The new edits read near zero, at or below
their least-squares values. The scaled split reproduces almost all of the gain edits' change but only half of the
rebuilt edit's. Part of rebuilt's lower value may therefore come from the split rather than the edit.

The sections below up to "Surprises" describe the first run; the second run has its own section after them.

## What we were trying to test

The 67M model has a parameter decomposition, VPD, whose subcomponents each hold a rank-one piece of a weight matrix. A
fine-tune changes the weights by some $\delta W$. To read that change in terms of the subcomponents, $\delta W$ is split
into a change of each subcomponent's write vector and read vector, plus a remainder that goes outside the
subcomponents. Two splits were compared:

- the **least-squares split**, the smallest change of the subcomponents that reproduces $\delta W$ as closely as
  possible, with the rest in the remainder;
- the **unscaled adjoint split**, $dU = \delta W V$ and $dV = \delta W^\top U$, whose weight change is
  $K(\delta W) = U U^\top \delta W + \delta W V V^\top$. The remainder takes $\delta W - K(\delta W)$, so the pieces
  still add up to the fine-tuned weights. `GOALS.md` defines the adjoint split with a per-matrix scale $c$ fitted so
  that $c K(\delta W)$ best matches $\delta W$ on the task's inputs; this run fixed $c = 1$.

The quantity read is the **gain share**: over the variables the behaviour needs, the share of their change in what
they write that a single gain on their original write explains. It reads 1 when every needed change is a rescaling
of an existing part, and 0 when none of it is. The needed variables are found by the same search as before, run
again under each split.

The seven edits were built by hand so that their answer is known. Two gain edits scale existing copy neurons by
two and by ten, and should read near 1. A rebuilt edit removes copy neurons and rebuilds the same function. Two new
edits add parts the model did not have, one with a few new neurons and one thin, and should read near 0. Two random
changes, one matched in size and one matched in its effect on the output, are the floor. The do-nothing, the model
compared with itself, must read exactly 1.

## What we expected, and why

The card filed two rivals. If the unscaled adjoint split carries each edit onto the subcomponents more faithfully, the
gain edits should move toward 1 and the new edits should stay at or below their least-squares values. If the split
does not matter, every edit should land within the search's measured noise of a few thousandths of its least-squares
value.

## What we observed: every built edit reads lower, and rebuilt no longer stands apart from new

No figure was made for this cell; the table is the result.

| built edit | least-squares split | unscaled adjoint split |
|---|---|---|
| gain edit, x2 | 0.761 | 0.207 |
| gain edit, x10 | 0.761 | 0.207 |
| rebuilt | 0.623 | 0.012 |
| new, few neurons | 0.013 | 0.013 |
| new, thin | 0.019 | 0.013 |
| random, size-matched | 0.011 | 0.0018 |
| random, output-matched | 0.011 | 0.0016 |

**Reading it.** Neither rival holds. The gain edits fell by more than half under the unscaled adjoint split, instead of
rising toward 1. The rebuilt edit lost almost all of its share and now reads like the two new edits. The new edits and
the randoms stayed low or went lower, so the gap between the floor and the new edits is no wider. The gain edits still
read above everything else, so the top of the ordering survives, but rebuilt can no longer be told apart from new.

The needed variables are not what moved. For the gain edits the search found the same set under both splits, so the
drop comes from how much of each variable's write change a gain explains, not from which variables were counted.

## Likely cause: without the scale, the adjoint pieces are large and cancel

This is the likely cause and was not checked further. Under the unscaled adjoint split, the change the subcomponents
write is far larger than $\delta W$ itself, and the remainder carries most of it back with the opposite sign. For the
x2 gain edit, as shares of $|\delta W|^2$ summed over the 24 matrices:

| part | share of the squared weight change |
|---|---|
| write side | 42 |
| read side | 8.5 |
| remainder | 60 |
| write and read cross term | 22 |
| cross term with the remainder | −131 |

These sum to 1, as they must. Under least squares the remainder share of the thin new edit is zero to rounding, so
nothing cancels there. On every edit the unscaled adjoint remainder alone is between 4.8 and 60 times the real change.
If the subcomponents' write change is mostly a large term that the remainder undoes, a gain on the original write can
explain little of it, which would lower every edit at once, as seen. The scale $c$ shrinks $K(\delta W)$ toward
$\delta W$, so it targets exactly this cancellation.

## What we learned

The adjoint split without its per-matrix scale does not give a cleaner gain share than least squares on any built
edit, and least squares stays the split under which the gain share is read. This does not test the adjoint split as
`GOALS.md` defines it: the cancellation seen here is what its scale is meant to remove. The second run, below, reads the scaled split.

## How compelling is the evidence

The apparatus held. The do-nothing read exactly 1 under both splits with no changed variable. Swapping the model's
own activations in under the adjoint featurization changed no prediction on the x2 edit. A fresh least-squares
reading over the earlier run's saved variable sets reproduced all seven earlier values to better than one part in ten
million, and the thin new edit's rescan found the same set. Each edit is one built state read once, on one
decomposition seed and one task. The drops on the gain and rebuilt edits are far beyond the search noise; the changes
on the new edits and randoms are small.

## Surprises

The gain edits went down, not up. Neither rival predicted that; the card filed it as "reported as read".

## Run 2: the adjoint split with its scale

### What changed

The second run repeated the first run's adjoint readings on the same seven edits with one change. Each matrix's adjoint
weight change $K(\delta W)$ was multiplied by its scale $c$, as `GOALS.md` Terms define the adjoint split. The scale is
the one number per changed matrix that makes $c K(\delta W)$ closest to $\delta W$. The distance is weighted by
$\Sigma$, the second moments of that matrix's inputs on the edit's own read rows. The remainder takes $\delta W - c K(\delta W)$, so the
pieces still add up to the fine-tuned weights. The needed variables were searched again under the scaled split, with
the first run's settings.

A second quantity is read beside the gain share. The **reach** $R$ of a matrix is the share of $\delta W$'s
$\Sigma$-weighted energy that $c K(\delta W)$ reproduces. It reads 1 when the scaled adjoint change is the whole weight
change, and 0 when it reproduces none of it. Each edit gets the median over its changed matrices. A low reach means
most of the edit sits in the remainder, outside the subcomponents.

### What we expected, and why

The scale was expected to lie between 0 and 1 on most matrices. The first run's unscaled pieces on the x2 edit carried
42 and 8.5 times the squared weight change, so the scale should shrink them toward the real change. Two rivals were
filed in `predictions_run2.json`. Under rival a, the scaled split separates the edits more cleanly than least squares.
Its gain edits would then sit more than the scan noise of 0.003 above their least-squares 0.761. Its new edits and
randoms would sit at or below their least-squares values. Under rival b as filed, the gain edits read at or below
0.761, anywhere down to the unscaled 0.207. The research-log card stated rival b more narrowly: the scaled split reads
like the unscaled one, within a narrow band of it. No edit had a verdict rule, since no reuse definition has been adopted yet.

### What we observed: the scaled split gives the least-squares order, at lower values

No figure was made for this cell; the table is the result. The reach is the median over each edit's changed matrices.

| built edit | least-squares split | unscaled adjoint split | scaled adjoint split | reach of the scaled split |
|---|---|---|---|---|
| gain edit, x2 | 0.761 | 0.207 | 0.713 | 0.97 |
| gain edit, x10 | 0.761 | 0.207 | 0.713 | 0.97 |
| rebuilt | 0.623 | 0.012 | 0.453 | 0.51 |
| new, few neurons | 0.013 | 0.013 | 0.0031 | 0.28 |
| new, thin | 0.019 | 0.013 | 0.013 | 0.44 |
| random, size-matched | 0.011 | 0.0018 | 0.0015 | 0.41 |
| random, output-matched | 0.011 | 0.0016 | 0.0014 | 0.42 |

**Reading it.** With its scale, the adjoint split puts the seven edits in the same order as least squares. The gain
edits read 0.71, against 0.76 under least squares. The rebuilt edit reads 0.45, against 0.62. The new edits read 0.003
and 0.013, at or below their least-squares values. The two randoms read about 0.0015, below their least-squares 0.011.
Without its scale, the rebuilt edit read as low as the new edits: 0.012 against 0.013. With it, rebuilt sits well
between the gain edits and the new ones again.

**The reach.** The scaled split reproduces 0.97 of the gain edits' change, on their one changed matrix. It reproduces
only 0.51 of the rebuilt edit's, as the median over five matrices. Part of rebuilt's fall from 0.62 to 0.45 may
therefore come from the split's limited reach. This run did not separate that from the edit itself. The new
edits and randoms reach between 0.28 and 0.44, so most of their change sits in the remainder.

**The scale and the cancellation.** The gain edits' one matrix got a scale of 0.125: an eighth of the unscaled
change. On the rebuilt edit's five matrices it ran from 0.019 to 0.19. With the scale, the
remainder's share of the x2 edit's squared change fell from 60 to 0.40. Its cross term with the subcomponents' change
fell from −131 to −0.54, so the large cancellation of the first run is gone.

**Which variables were counted.** Held at the least-squares run's needed variables, the gain and rebuilt edits read
the same values as with their own. Their fall from least squares therefore comes from the split, not
from a different set of variables. On that held set the few-neuron new edit reads 0.0050, and the output-matched random
reads 0.0027.

### What we learned

The adjoint split as `GOALS.md` defines it, with its per-matrix scale, orders the built edits as least squares does.
The gain edits read highest, rebuilt next, and the new edits and randoms near zero. Every value is lower than under
least squares, and rebuilt's lower value may partly reflect a split that reaches only half of its change. The scale is
what keeps rebuilt apart from the new edits: without it, rebuilt read at their level. This run gives readings only and
does not choose between the two splits.

### How compelling is the evidence

The apparatus held on all four pods. The do-nothing read exactly 1 with no changed variable. Swapping the model's own
activations in under the scaled featurization changed no prediction on the x2 edit. Its largest logit change was
5.1e-5, as in the first run. The pieces plus the remainder matched $\delta W$ to a relative 3e-17, under the filed bar
of 1e-12. Every fitted scale was positive.

Each edit is again one built state read once, on one decomposition seed and one task. The scan noise is 0.0026, the
spread of the gain share between two scan seeds on one built edit. The gain and rebuilt edits' falls from least squares
lie far beyond it, while the new edits' falls are a few times it. The reading is blind to whether the base weights the
input statistics were read on match each edit's own; that per-item check was not run.

### Surprises

None in direction: the outcome lies inside rival b as filed. The gain edits landed near their least-squares value
rather than near the unscaled one, which the card's narrower rival b did not expect.

## What next

- The per-item weight check listed under "Not run", before any of these values goes into the write-up's answer.
- Whether rebuilt's lower value under the scaled split comes from its reach of 0.51 or from the edit. That needs a new
  cell from the user.

## Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| R0a | do-nothing reads exactly 1 under each split | stop unless 0 changed, dd 0, gain share 1 | 0 changed, 1 under both | held |
| R0b | identity swap under the adjoint featurization changes no prediction | stop on any argmax change | none | held |
| R0c | least squares reproduces the earlier run | stop unless within the filed relative tolerance | all seven within it | held |
| rival a | unscaled adjoint separates more cleanly | gain edits nearer 1, new edits at or below | gain edits fell | not met |
| rival b | split does not matter | every edit within scan noise | gain and rebuilt far outside | not met |
| run 2 R0a | do-nothing reads exactly 1 under the scaled split | stop unless 0 changed, dd 0, gain share 1 | 0 changed, 1 on all four pods | held |
| run 2 R0b | identity swap under the scaled featurization changes no prediction | stop on any argmax change | none | held |
| run 2 rival a | scaled adjoint separates more cleanly than least squares | gain edits above 0.761 by more than 0.003; new edits and randoms at or below least squares | gain edits 0.713 | not met |
| run 2 rival b, filed | gain edits at or below least squares | gain edits at or below 0.761, down to 0.207 | 0.713 | met as written |
| run 2 rival b, card | scaled reads like unscaled | gain edits within a narrow band of 0.207 | 0.51 above it | not met |

## Departures

- From `GOALS.md` Terms: the adjoint split there carries a per-matrix scale fitted on the data. This run used it
  unscaled, as filed in `predictions.json` under "split_choice". That choice is the likely cause of the result.
- Run 2 used the adjoint split with its scale, as `GOALS.md` Terms define it: no departure.
- From the literature: none.

## Not run

- A per-item weight check: that the base weights the input statistics $\Sigma$ were read on are the ones each edit was
  built on. It was recommended before run 2 and not added mid-run.
- A separation of rebuilt's lower scaled value into the split's reach and the edit itself.
- An independent recompute of run 2's numbers; they are not in the write-up's answer paragraph.
- Any check of the first run's cancellation beyond the shares above.
- A figure: the cell was asked for none.

## Provenance

Results `experiments/adjoint_rho/results/items_s0.jsonl` to `items_s3.jsonl`; stage 0 holds the controls and the
least-squares reproduction, stage 1 the thin new edit's rescan, stage 2 the adjoint readings. Code
`experiments/adjoint_rho/run.py`, split code `src/commute/kstar_vpd.py`. Predictions
`experiments/adjoint_rho/results/predictions.json`. Four RunPod H100 pods, one shard each, results commit 31be0ad9.

Run 2: results `experiments/adjoint_rho/results/items_run2_s0.jsonl` to `items_run2_s3.jsonl`, logs `run2_s0.log` to
`run2_s3.log`; stage 0 holds the controls, stage 2 the scaled readings. Predictions
`experiments/adjoint_rho/results/predictions_run2.json`. Four RunPod H100 pods, one shard each, about 1,450 s per long
item, results commit 41b087a3. Results check `reports/reviews/2026-11-27_review_e331-results.md`.
