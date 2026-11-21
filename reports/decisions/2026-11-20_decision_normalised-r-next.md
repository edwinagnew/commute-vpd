# Size-normalising R gets the built edits right but still follows update size on trained pairs: what next?

R is the share of the parts the behaviour runs through that still compute what they did in the base model. N is the same, but each part's change is counted per unit of that part's weight change.

**What we ran**

- Weighting parts by how much the behaviour depends on them. We expected it to fix the near-1 scores. It made things worse: a ten-fold-gain reuse edit scored 0.938, a thin new edit 0.997. ([report](../analyses/2026-11-21_analysis_p1-42-r-reanalysis.md))
- Dividing 1 - R by update size squared. It orders the six built edits but overcorrects: the reuse side is higher on only 4 of 9 pairs. ([results](../../experiments/r_reanalysis/results/crude_norm.json))
- N. We expected every built reuse edit above both new ones, and it held: ten-fold gain 0.23 against -2.04 and -3.21. But N is not a 0-to-1 share (most values are below 0), and on trained pairs the reuse side is higher on only 5 of 9, with bigger updates reading more reused. ([results](../../experiments/kstar_vpd_norm/results/summary.json))

**So:** normalising fixes the built edits, but every version so far still follows update size on the trained pairs.

**Question 1.** What next for R?

- (a) Recommended: one cheap cluster run testing N on built reuse and new edits made at the same update size. If N still separates them, size is not what it reads, and the trained pairs' labels are what is in doubt.
- (b) Adopt N for the paper, with its limits stated.
- (c) Keep R as worded and report that such scores follow update size.

Only you change the definition's wording.

**Question 2** (still open from the last card). The memorising twins:

- (a) Recommended: say they check how the pairs were built, not R.
- (b) Keep them as a case R must read low on.
- (c) Drop them.
