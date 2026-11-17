# Do the other readings follow the share on the four 96-key pairs?

*E147, cell P2.12, compare-lead, 2026-11-10. Plan `reports/plans/2026-11-10_plan_e147-refit-96key.md`, predictions
`experiments/refit_96key/results/predictions.json`, results `experiments/refit_96key/results/stage{0,1}.json`. Sides are named as
each pair was built; no verdict of reused or new.*

We fine-tuned E58's eight 96-key maps again at their original recipe and seeds and kept the updates this time. The 24 refits
reproduced E58's saved shares well inside each map's seed spread. On these updates the re-gating share separates three of the four
pairs, as filed. None of the activation readings or the untrained decomposition matches it: each separates at most one pair, and two
of those separations point the other way. So the 96-key rows of part 2's table now look like the other built pairs: only the share
follows the build on all three.

## What we were trying to test

E133's method table compares the re-gating share with simple readings on pairs of fine-tunes. Each pair is built so that one side
should use an ability the model already has and the other should need a new one. Four pairs from E58 had no kept updates on both sides, so
five of their cells were empty or held an older reading:

- **rank-1 against a cipher**: a map whose answers follow the model's own top guess, against a random bijection.
- **copy against a prior-matched cipher**: a copy map, against a cipher picked so the original model gives its targets the same prior.
- **half-known against its shuffle**: half-known facts, against the same values moved to other keys.
- **class map against its twin**: one word per class of key, against the same words dealt out with no class structure.

Each map has 96 keys. We fine-tuned it for seeds 0, 1 and 2, with E54's recipe exactly as E58 called it, and read the update at
step 30. Each update was saved in bf16, and every reading below is of the saved file. The readings are shares of the update's own gain
in log-odds of the target. A share of 0 means the reading carries none of the gain; 1 means it carries all of it.

- **re-gating share**: how much of the gain comes from which subcomponents Γ switches on, read on the trained decomposition.
- **untrained decomposition**: the same share on each matrix's SVD, a decomposition that needs no training.
- **two neuron forms from E98**: the original MLP output weights read on the update's firing, either at every neuron (whole width)
  or with only the MLP input changed (half).
- **TopK neurons from E143**: each block's MLP kept on its top neurons, at 64 per block and at the decomposition's on-count.
- **fair steering from E143**: pooled and per-key vectors, with the strength and layer chosen on training rows.

Two controls ran first, alone. First, each reader read a state it had read before, to check that this job's path matches theirs. Second,
each refit's share was checked against E58's saved share of the same map and seed. The bar was that map's seed spread, from 0.0015
for copy to 0.045 for rank-1.

## What we expected, and why

We expected every refit to land within about 0.0001 of E58's share, because training on this card is bitwise repeatable and bf16
storage adds about that much. For the comparison we filed the 24-key versions of the same maps as our guide. There the untrained
decomposition, the two neuron forms and fair steering each separated fewer pairs than the share. The TopK neurons wrote almost nothing
on any pair. We filed the same here: the share separates rank-1, copy and half-known from their partners but not the class map from
its twin. Every other reading separates fewer pairs, and TopK reads as inert.

## What we observed

### Do the refits reproduce E58's updates?

**Reading it.** Yes. All 24 refit shares fell within their map's bar. The largest difference was 0.0006, on rank-1 seed 1, against a
bar of 0.045. The seven updates an earlier run had kept were identical to their refits, with a largest weight difference of exactly 0.
The readers' own controls also read 0 difference against their saved readings, and holes_fill's stage 0 passed unchanged. So the
readings below are of E58's own updates, not of near copies.

### Does any other reading separate these pairs the way the share does?

Gap = side built to reuse minus side built to be new, in shares of the gain, mean over three seeds a side. "Separates" needs a gap of
at least 0.1 and larger than either side's seed range, which is E143's rule.

| reading | rank-1 vs cipher | copy vs prior-matched cipher | half-known vs shuffle | class map vs twin |
|---|---|---|---|---|
| re-gating share | 0.56, separates | 0.39, separates | 0.25, separates | -0.03 |
| untrained SVD | 0.02 | -0.16, separates | -0.08 | -0.01 |
| neuron, whole width | -0.02 | -0.13, separates | 0.03 | -0.04 |
| neuron, half | -0.03 | -0.05 | 0.02 | -0.02 |
| TopK neurons, k 64 | -0.02, inert | 0.01, inert | -0.01, inert | -0.01, inert |
| steering, pooled | 0.03 | 0.59, separates | 0.04 | -0.07 |

The redrawn table is `experiments/method_table/figures/final.png`.

**Reading it.** No reading follows the share on these pairs. The share separates three of the four pairs. Steering separates one, copy
against the prior-matched cipher, with a gap of 0.59 and seed ranges near 0.03. The untrained decomposition and the whole-width neuron
form also separate only that pair, and in the other direction: the side built to be new reads higher there. The half-width neuron form
separates none, and the TopK neurons carry under 0.05 of the gain on every state. On the class map, the share gap is 0.03 below zero
and no reading separates the two sides.

## What we learned

The four 96-key rows now read like the other built pairs in E133's table. The re-gating share follows the build. On these four pairs steering
follows it only on the copy map, the one map whose target sits in the window. The neuron readings and the untrained decomposition do not
follow it. This is the comparison part 2 of the plan needs, and it no longer rests on the 24-key versions of these maps alone.

## How compelling is the evidence

Each pair has three seeds a side. On E135's spread, the bar of 0.1 is about four standard errors. Every cell was read on the same 24
saved updates, on the held-out windows of each map's trained keys, the rows E58's share was read on. All readings are blind to anything
that does not move the target's log-odds at the answer position. TopK is inert here as on every other pair, so its column cannot tell
pairs apart either way. The verdicts compare readings against the share; they do not say which side reuses anything.

## Surprises

None against the filed predictions: all four share calls and all five verdicts read as filed. Two changes to the table are worth
naming. First, on rank-1 against the cipher and on half-known against its shuffle, the older one-vector steering had shown the side
built to reuse higher. The fair form shows no difference there. Second, the class-map row of the share now reads no clear difference.
Before, it read apart on E136's two regenerated seeds under the seed-range rule, at a gap of about the same size.

## What next

- Read effective rank on the 24 kept updates with E141's reader: it is the one column of these rows still empty. If it separated the copy pair
  in the share's direction, the rank column would join steering as a partial follower.
- Nothing else on these pairs: the card's stop holds, since each family separates fewer pairs than the share and none the share does not.

## Scorecard

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| reproduce | each refit's share matches E58's | within the map's seed range | largest difference 0.0006 | as filed |
| share calls | share separates rank-1, copy, half-known; not the class map | E143 call | separates, separates, separates, does not | as filed |
| untrained | fewer pairs than the share | E143 compare | 0 of 3, one opposite | rival B, as filed |
| neuron | fewer pairs than the share | E143 compare | 0 of 3, one opposite | rival B, as filed |
| neuron_half | fewer pairs than the share | E143 compare | 0 of 3 | rival B, as filed |
| neuron_top | inert | E143 compare | inert on all four | no call: inert, as filed |
| steering | fewer pairs than the share | E143 compare | 1 of 3 | rival B, as filed |

## How the method table changed

`experiments/method_table/plot_final.py` gained a reader, `e147_refit`, that runs last. For these four pairs it supplies the share,
untrained, the three neuron forms and steering, each with E147's filed call.
- **E58's saved share** and, for the class map, E136's two regenerated seeds are replaced by the refit share. Their sources are kept in
  each cell's `replaces` field.
- Why replace rather than keep both: the refit reproduced E58's share, so every cell of a row now reads one set of updates.
- **The older one-vector steering** is replaced by the fair form.
- **Prior and size** are unchanged.

`final.json`, `final.csv` and `final.png` are redrawn.

## Departures from the literature

None beyond those of the readers reused here, E98, E136 and E143, each stated in its own report. The training recipe is E54's, unchanged.

## Not run

- Effective rank on the refit updates.
- Cross-application and cosines of the steering vectors, which are not in E133's cells.
- The share at the gate step or at step 10.
- New shuffle seeds: three already exist.
- Any verdict of reused or new.

## Provenance

Job 57961 on a debug L40, about 13 minutes, $0. The first job, 57955, failed only because `results/large` was not linked. The
updates are `experiments/refit_96key/results/large/dW_<map>_s<seed>.pt`, 24 files in bf16, sha256 in `stage0.json`.
