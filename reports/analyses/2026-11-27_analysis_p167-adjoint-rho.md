# Does the adjoint split without its scale separate the built edits more cleanly than least squares?

We read the gain share on seven hand-built edits, splitting each weight change over the subcomponents by least squares
and by the adjoint split without its scale. Without its scale the adjoint split separated the edits worse: the rescaled
edits fell from about three quarters to one fifth, and the rebuilt edit fell to the level of the new ones. Least
squares stays the split for the gain share, and the adjoint split as `GOALS.md` defines it, with its scale, is untested.

## The answer

Splitting each built edit's weight change by the unscaled adjoint split instead of by least squares makes every edit
look less like rescaling, not more. The two gain edits fell from 0.761 to 0.207. The rebuilt edit fell from 0.623 to
0.012, next to the new edits at 0.013. The likely cause, not checked here, is that the unscaled split writes a change
many times the real one onto the subcomponents and cancels it in the remainder. Correcting that is the scale's job.

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
`GOALS.md` defines it: the cancellation seen here is what its scale is meant to remove, so the scaled split is open.

## How compelling is the evidence

The apparatus held. The do-nothing read exactly 1 under both splits with no changed variable. Swapping the model's
own activations in under the adjoint featurization changed no prediction on the x2 edit. A fresh least-squares
reading over the earlier run's saved variable sets reproduced all seven earlier values to better than one part in ten
million, and the thin new edit's rescan found the same set. Each edit is one built state read once, on one
decomposition seed and one task. The drops on the gain and rebuilt edits are far beyond the search noise; the changes
on the new edits and randoms are small.

## Surprises

The gain edits went down, not up. Neither rival predicted that; the card filed it as "reported as read".

## What next

- The scaled adjoint split, with $c$ fitted per matrix on the task's inputs by `split.adjoint_scale`. If the gain edits
  then read at or above their least-squares value with rebuilt apart from new, the adjoint split is worth keeping.
- Run 2 of the cell: its design is with the user.

## Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| R0a | do-nothing reads exactly 1 under each split | stop unless 0 changed, dd 0, gain share 1 | 0 changed, 1 under both | held |
| R0b | identity swap under the adjoint featurization changes no prediction | stop on any argmax change | none | held |
| R0c | least squares reproduces the earlier run | stop unless within the filed relative tolerance | all seven within it | held |
| rival a | unscaled adjoint separates more cleanly | gain edits nearer 1, new edits at or below | gain edits fell | fails |
| rival b | split does not matter | every edit within scan noise | gain and rebuilt far outside | fails |

## Departures

- From `GOALS.md` Terms: the adjoint split there carries a per-matrix scale fitted on the data. This run used it
  unscaled, as filed in `predictions.json` under "split_choice". That choice is the likely cause of the result.
- From the literature: none.

## Not run

- The scaled adjoint split (`split.adjoint_scale`), with its fitted scale not equal to 1.
- Run 2 of the cell: its design is with the user.
- Any check of the cancellation's cause beyond the shares above.
- A figure: the cell was asked for none.

## Provenance

Results `experiments/adjoint_rho/results/items_s0.jsonl` to `items_s3.jsonl`; stage 0 holds the controls and the
least-squares reproduction, stage 1 the thin new edit's rescan, stage 2 the adjoint readings. Code
`experiments/adjoint_rho/run.py`, split code `src/commute/kstar_vpd.py`. Predictions
`experiments/adjoint_rho/results/predictions.json`. Four RunPod H100 pods, one shard each, results commit 31be0ad9.
