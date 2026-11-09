# Decision: how to read steering and the neuron version on a real fine-tune and its shuffled-label twin

*compare-lead, 2026-11-09, for the pending-decisions card. Blocking this comparison, due 11-11. Nothing has run.*

**The question.** The comparison was to give steering a fair chance with one vector per word, and to tell apart a fine-tune taught a real label rule from its twin, whose labels are shuffled word by word. Two things stop it as written. One vector per word cannot be fitted on the rows the share was read on, and two seeds per side are too few for the rule that says a run can resolve the difference we care about. What should run?

**The evidence.**

- The share was read on the first 256 labelled windows of each pair. On those rows, only 15 of 128 scored windows on the case-label pair, and 34 of 128 on the repeat-label pair, have a window of their own word in the half used for fitting. A vector for a word with no fitted window does not exist.
- Each pair also has training windows, about 16 per word, which cover every word on those rows. All 1024 labelled windows would cover 205 of 512 and 430 of 512 scored windows.
- The smallest difference that would change what we do is 0.1 of the gain; the seed spread saved on these rows is 0.056; with two seeds per side that is 1.80 standard errors, under the bar of 2, and three seeds per side would pass. A third seed is a new fine-tune, which the current brief rules out.
- The other six changes the user asked for check out against the code and fit in one run.

**What could go wrong.** Fitting on training windows uses windows the fine-tune learned from, so a per-word vector may read higher than one fitted on unseen text. The share is unaffected, since it is not fitted. The twin's norm is about three times the real update's on the case-label pair, which the run reports by rescaling.

**The options.**

- **Which windows the per-word vectors are fitted on.**
  - (a) Training windows, about 16 per word; scored on the same 256 rows as the share. Every row covered, no new share read needed. Limit: the fitting windows are ones the fine-tune saw.
  - (b) All 1024 labelled windows, even ones fitted and odd ones scored; only words with a fitted window are scored; the share is read again on those rows. Limit: covers 40% of the case-label rows.
  - (c) One vector per label instead of per word. Every row covered, but the vector is picked by each window's own answer, so it hands steering the answer on both sides.
- **The seed count.**
  - (x) Read at two seeds per side as a description, not a test, as the brief says for "beyond seed spread". Free, about ten minutes on the free queue. The run's card is refused, so it needs an override of the card rule from the user.
  - (y) Add a third seed of each arm of both pairs as a new piece of work: four more fine-tunes, a few minutes on the free queue. It passes the bar, at the cost of one new fine-tune of each kind.
  - (z) Close the comparison not run.

**What happens next.** Under (a) and (x), the code writer builds this comparison next, and it runs in a few minutes; the result is a figure with one paragraph, no verdict. Under (z) the part of the paper that asks whether any activation reading sees the difference between a real fine-tune and its twin stays open.

**Recommendation.** (a) and (x): every row covered, and the description the brief already asks for.

*Names in the reports: this is cell P2.4, card `reports/plans/2026-11-09_plan_p2-4-card.md`, the user's item 1 and item 6.*
