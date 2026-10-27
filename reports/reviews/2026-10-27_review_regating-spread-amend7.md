# Review: E36 Amendment 7, is the original values' edge only each value's room to move? (2026-10-27)

Independent check of `experiments/regating_spread/amend7.py`, its filed predictions
(`results/predictions_amend7.json`), its results (`results/regating_spread_amend7.json`, `results/large/amend7.log`)
and the plan (`reports/plans/2026-10-26_plan_regating-without-cues.md` → Amendment 9). The reviewer did not write the
code or the analysis and ran nothing; numbers were read with `jq`.

**In short.** I re-scored the run from its results file, re-read the two value-bin rules against their description, and
compared their path through the evaluation with the nearest rule's. The score is right and nothing in the code gives
the nearest rule an unfair edge: predicting each subcomponent's change from its own original value alone recovers at
most a quarter of what the nearest rule on all the original values recovers. That settles the narrow question, but
not the threat it was built to close, because a rule that joins the text summary with each value's own room to move was
never run, so "the model's view beats the text only through the bounds" is still open.

## 1. Scoring (correct under the filed rule)

| Check | Filed rule | Step 10 / 100 / 300 | Verdict |
|---|---|---|---|
| A0 | near_g0 within 0.02 of Amendment 6 | 0.886 / 0.776 / 0.762 against 0.890 / 0.779 / 0.762 | holds |
| C_bin, bin16q | R ≥ R_fix − 0.05 | 0.546 / 0.299 / 0.176 against fix 0.495 / 0.202 / 0.024 | holds |
| C_bin, bin16w | same | 0.562 / 0.270 / 0.123 | holds |
| C_fit, bin16q | fitted MSE ≤ fix's | 0.001167 ≤ 0.001186; **0.001631 > 0.001628; 0.001718 > 0.001707** | fails at 100 and 300 |
| C_fit, bin16w | same | 0.001128 / 0.001573 / 0.001654, all below fix | holds |
| readable | one bin rule passes both at every step | bin16w | readable |
| B1 | max(bin16q, bin16w) ≤ near_g0 − 0.3 at step 300 | 0.176 ≤ 0.462 | **holds**, margin 0.29 |

The JSON's `score` (`passing_controls: ["bin16w"]`, `best_bin_R` 0.1763, `B1: true`) matches this, and so does the
log. The scoring code (`amend7.py:228-240`) does what the filed rule says: readability needs one rule passing both
controls at every step, and B1 then reads the best R over both rules, the failing bin16q included. The code and the
predictions file are unchanged since commit `51b9aa2` (filed 10:54), and the run started at 10:58.

## 2. Are the bin rules what the docstring says, and is the comparison fair? (yes)

- **Edges.** Equal-count edges are the sorted fitted values at ranks iN/16 − 1 (`amend7.py:168,174`); equal-width edges
  are i/16 (`:169,174`). `searchsorted` with its default left side gives the number of edges strictly below the value
  (`:143,149`), so a bin holds e_b < v ≤ e_{b+1}, as documented. Γ's values are clamped to [0, 1] in the forward pass
  (`param_decomp/models/sigmoids.py:23`), so exact zeros are common and fall in bin 0 under both rules.
- **Empty bins** take the nearest non-empty bin below, else above (`:153-155`); a constant column lands in one bin and
  every other bin copies it. Bin 0 of the equal-count rule is never empty.
- **Fitted against held-out.** Edges, bin means and the line are fitted on the 262,144 positions of `train[:512]`
  (`:220-225`); every R is read on `held` (`:196-208`). Amendment 4 measured no 16-token span shared between the two.
- **Same path for every rule.** Each rule adds its change to the held-out position's own original values and clips to
  [0, 1] (`:205-208`); `fix` clips a·g0 + b, as Amendment 1 did. All go through the same `S.with_alive` and
  `S.masked_logp` call (`:211`), share one g0 and one g1 reference, and are normalised by one gap (`:215-216`).
- **No held-out information reaches near_g0.** Its neighbours are found from the held-out position's original values
  only (`:202`); the updated values `ci1` enter only the g1 reference. Column order agrees: the bin tables are
  concatenated chunk by chunk in the same column order as `to_flat` (`:172-185`), and `lookup` is the same function at
  fit and at evaluation (`:179`, `:206`).
- **Mean squared error** is computed identically for the bins and the line, on the change as applied, and divided by
  the same count (`:158-160`, `:179`, `:183`, `:186`).

The earlier code review tested `fit_columns` against a brute-force rule on CPU, ties and empty bins included, and found
agreement to 1e-7 (`reports/reviews/2026-10-27_review_amend7-code.md`, "Checked and found correct"). I did not repeat
that test; my reading agrees with it.

## 3. Findings, by severity

### 3.1 Medium-high (reading): B1 tests room to move as the whole story, not as the source of the edge over text

The threat came from Amendment 6's review (`reports/reviews/2026-10-26_review_regating-spread-amend6.md`, section 4,
"The original values have a mechanical edge on the clip"). It was about the gap between the original values and the
text summary: neighbours with similar original values respect the query's bounds, and text neighbours need not. If
that were the whole gap, then text neighbours made aware of each value's room would match the original values.

B1 asks something narrower: can each value's own room, with no context at all, match the original values? It cannot.
The best bin rule reads 0.18 against 0.76. But the bin rules and the text rule carry different information. The bin
rules know the bounds and nothing about context, and the text rule knows context but not the bounds. A rule that uses
both was not run. One example adds the bin rule's change plus the text neighbours' leftover change, d minus their bin
means. Another averages, for each subcomponent, only text neighbours in the query's value bin.

Adding the two recoveries gives 0.70, near the 0.76 of the original values. Recovery is not additive, so this shows
only that such a rule could plausibly close the gap, not that it would.

So the decision tree's sentence "the original values' edge is not each value's room to move" (`amend7.py:123`) and the
log card's "the restatement holds only if that stays at least 0.3 below" (`docs/research_log/parts/sections/decisions.html:7`)
read more into B1 than it holds. What B1 supports: a function of each subcomponent's own original value recovers at
most about a quarter of what the nearest rule does, so the nearest rule's success needs joint structure across
subcomponents. What it does not yet support: that the original values beat the text summary for any reason beyond the
bounds. The plan's restatement, "positions the original model treats alike get alike changes", has this rival:
"similar text gets similar changes, within each value's room". B1 does not tell the two apart.

### 3.2 Medium-low: neither bin rule is the strongest per-value rule

The median top equal-count edge is exactly 0 at every step (`regating_spread_amend7.json`, `top_edge_q_median`). For at
least half of the subcomponents, 15/16 or more of the fitted positions sit at exactly 0. bin16q is then two bins for
them: exactly zero, and any positive value. bin16w resolves the positive range but puts zero in one bin with values up
to 1/16. No rule run has both an exact-zero bin and fine bins above it. Its R is unknown.

The margin makes a reversal unlikely. To fail B1 a per-value rule would need 0.46, and bin16q and bin16w differ by only
0.05. But this is an argument, not a measurement.

A second gap: each bin predicts the mean change, which minimises squared error but not the masked model's loss. The
rankings disagree. bin16w has the lowest fitted error, 3% below the line, but bin16q has the higher R. The line's error
lies between theirs, yet its R is 0.02. At positions where a subcomponent is exactly zero, a mean rule switches every
such position on a little. The nearest rule averages 8 real, sparse change vectors and keeps most entries at zero. A
per-value rule that kept the change sparse at zero values was not run. Examples are the bin median, or no change in the
zero bin. Whether sparsity alone is worth much of the gap is open.

### 3.3 Low: the controls certify the code more than the strength of the null

- C_bin has no power at step 300: with the line at 0.024, any bin rule above −0.026 passes. At step 10 it has some.
- C_fit is the working control. bin16w passes it by 3% at step 300. bin16q fails it by 0.2% and 0.6% at steps 100 and
  300. That is the expected cost of its single positive bin, which the code review predicted, not a bug. It changes
  neither readability, since bin16w passes, nor the score, since B1 reads the maximum and bin16q is the maximum.
- No control plants a known per-value change and checks that a bin rule recovers it in R. The implementation rests on
  the code review's brute-force test and on reading.
- The fitted error of zero change and the nearest rule's held-out error were not reported. So the share of the change's
  variance that a per-value rule explains is unknown, and the error numbers cannot be read on an absolute scale.

### 3.4 Low: a joint statistic short of "positions treated alike" is untested

Euclidean neighbours in the original values agree closely on how many subcomponents are on at a position, overall and
per matrix. A rule from each value plus one or two such activity counts is joint but coarse. If it recovered most of
the switching, the restatement would shrink to "positions with similar overall activity". The token rule, 0.33, and
the position rule, 0.04, were tested in Amendment 4. An activity count was not.

### 3.5 For the record (no action)

- A0's reference, Amendment 6, itself sits 0.004 from Amendments 4 and 7 at steps 10 and 100. Amendment 7 reproduces
  Amendment 4's near_g0 to 1e-14. Both are far inside the 0.02 tolerance and the 0.29 margin.
- The 4-token text rule with 32 neighbours reads 0.569 / 0.554 / 0.525, against Amendment 6's 8-neighbour 0.564 /
  0.525 / 0.511. The 8-neighbour cap costs the text rule at most 0.03, so it does not explain the 0.24 gap to the
  original values.

## 4. What the result is blind to

- Context combined with each value's bounds (3.1): the comparison that separates "the model's view" from "text plus
  room" was not made.
- Per-value rules other than two 16-bin mean tables (3.2): no exact-zero bin with fine bins above, no sparsity-keeping
  statistic.
- Coarse joint statistics (3.4).
- Which subcomponents or positions carry the nearest rule's edge: R is one sum over all held-out positions and all
  alive subcomponents.
- One fine-tune, of lowercased text; one seed; one held-out set of 64 rows; one 67M model; B1 is scored at step 300
  only.

## Not checked

- No numbers were recomputed from the model and no GPU or CPU run was made. The bin code was not re-tested on CPU; I
  relied on reading and on the code review's brute-force test.
- That the pod ran the committed `amend7.py`: I checked only that the code was unchanged after filing and that the log
  starts after the filing time, not a sha256 of the synced file.
- `amend1.task()` and `finetune_arm_b` were not re-read; the fine-tune is the one Amendments 4-6 used, and A0 holds.
- No lint or pyright on `amend7.py`.
- The combined rules named in 3.1, 3.2 and 3.4 were not run; whether any of them closes the gap is unknown.
