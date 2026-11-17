# Does the spectrum of the update alone tell a pair's two sides apart, as the share does?

*compare-lead, 2026-11-10, cell P2.9, E141. One CPU run on saved weight changes, no model and no forward pass, $0.
Sides are labelled as each pair was built; no verdict on reuse. Provisional, no independent check.*

For every saved fine-tune of nine task pairs I took the effective rank of its weight change, from its singular values
alone. The effective rank counts how many directions the change spreads over. I asked whether it tells each pair's two
built sides apart as the re-gating share does on the same fine-tunes. The rank separates five pairs and the share
seven, and they agree on only three. The rank misses both pairs whose sides have about the same update size. It
separates both label pairs the share does not, where the memorising twin trained three to ten times longer. So here
the update's spectrum is no substitute for the share: it looks more like a record of training length than of which
side was built to reuse, a cause this run does not test.

**The figure** is `experiments/update_svd/figures/gaps.png`. Per pair, it shows the gap between the side built to reuse
and the side built to be new, over the larger spread inside a side. Both readings use the same fine-tunes. Outside the
grey band the pair separates. On the two size-matched pairs the rank's gap is 0.07 and 1.6 against spreads of 6.7 and
11.4. The copy against random maps pair passes by a hair, a gap 1.03 times its spread, as it did when first read.

**Filed predictions** (`experiments/update_svd/results/predictions.json`), all three read as leaned:

| id | claim | read | verdict |
|---|---|---|---|
| P1 | the rank separates at least as many pairs as the share | 5 against 7 | rival B, the share separates more |
| P2 | the rank separates some pair the share does not | both label pairs | holds |
| P3 | where the rank separates, the reuse side is higher | higher on 3, lower on the 2 label pairs | does not hold |

The controls passed first: ten equal directions read 10, one reads 1, the zero update is undefined, and two of the
earlier run's fine-tunes read their old values exactly. The fine-tune that learned nothing reads 562, above every pair.

**Pairs covered and holes.** Saved seeds per side are listed in the plan, `reports/plans/2026-11-10_plan_E141-update-svd.md`,
from two to eight. Four pairs are holes, because one or both sides kept no weight changes. They are rank-1 against
cipher at 96 keys, copy against the prior-matched cipher, half-known against its shuffle, and the class map against
its twin. The step ladder was never run.

**Departures from the literature.** None: the effective rank is Roy and Vetterli's, averaged over matrices by squared
size as in the earlier weight-statistics run, and the separation rule is that run's.

**Not run.** The rank against update size and training steps as a cause; the other spectrum readings as scored columns;
the holes above; any Qwen pair.
