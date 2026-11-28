# Does the Qwen re-gating share change when a fine-tune is scaled down to a small answer gain? (cell P2.91, 2026-11-27)

We took the known-facts and copy fine-tunes of Qwen3-1.7B, three seeds each, and read their re-gating share on the VPD
decomposition at full strength and again with each update shrunk until it raised the trained answers by only about
one log-odds, the amount the random reference updates raise them, first on the original rows and then on the same
items packed into new rows. Shrinking the update raised the known-facts share by about 0.15 on both packings, while
the copy share fell a little on the original rows and barely moved on the new ones, and at that matched answer gain
the fine-tunes read within the random updates' range on known facts and below or level with the lowest draw on copy, never
above it. For the paper's goal this means the Qwen appendix's comparison of
full-strength fine-tunes with small random updates mixes how far an update moves the answer into what it says about
how the update was built, so on these two tasks it cannot separate a fine-tune from a random change.

## The answer

On known facts the share of the three fine-tunes rose from 0.58 to 0.60 at full strength to 0.72 to 0.74 when
scaled down to an answer gain of about one. On repacked rows the same reads gave 0.60 to 0.62 and 0.74 to 0.76.
On copy the share fell from 0.61 to 0.64 to 0.58 to 0.59 on the original rows. On repacked rows two seeds read 0.63
at gain one, and the third gave no reading because its gain fell under the floor of 1. Three random updates at gain
one, read on the original rows, gave 0.51 to 1.03 on known facts and 0.63 to 0.72 on copy. Repacking moved each
fine-tune's share by at most 0.05, and moved the one random update read again by 0.18.

## What was read

**The share.** For one update, the re-gating share is a ratio of two answer changes. The numerator is how much the
trained answers' log-odds rise in the masked model that keeps the base model's subcomponents but switches them on
and off as Γ reads them on the updated model. The denominator is how much the answers' log-odds rise in the updated
model itself, which we call the answer gain. Both are means over whole trained answers, with the end-of-response
token included (P2.90's function in `src/commute/share.py`, the same rows and answers as E328 and E330: 1,066 whole
answers on known facts, 1,732 on copy, on each task's 128 rows of 256 tokens). A share of 0 means changing which
subcomponents are on does nothing for the answer; 1 means it alone moves the answer as far as the update does.

**A share above 1.** The share is not a fraction and is not capped at 1. A value above 1 means that switching the
base model's own subcomponents on and off as the updated model would raises the answer further than the whole update
does. The rest of the update, the change to the weights themselves, then pulls the answer back down. Random draw 2
on known facts reads above 1 on both of its row sets.

**Scaling the fine-tune.** Each fine-tune update is the weight change $R = W_{\mathrm{ft}} - W_0$ on the
decomposition's own matrices, at the last save. We read the model $W_0 + cR$ for one factor $c$ between 0 and 1,
found by the same search E330 used for its random draws, until the answer gain sat in a band around a target.
The target was about 10 and about 1, beside the full update at $c = 1$. The bands were 9.55 to 10.45 and 1.003 to
1.097 log-odds, and every scaled read landed inside its band. Scaling keeps the update's direction fixed and changes
only how far it moves the model, so it tests whether the share depends on the answer gain for a fixed update.

**What scaling is not.** A scaled fine-tune is not a random update of matched size. To reach a gain of about one,
E330's random draws needed 8.8 to 23.3 times the fine-tune's norm, while the scaled fine-tunes sit at $c$ of 0.00875
to 0.0100 of it. The random draws also moved the model's whole output distribution much more. Their mean KL per
position was 0.0132 to 0.0662 on known facts and 0.0270 to 0.1123 on copy. The scaled fine-tunes at gain one read 0.0013
to 0.0014 and 0.0034 to 0.0035, about 9 to 53 times less on known facts and 8 to 33 times less on copy. Gain is
matched here; norm and side effects on other outputs are not.

**Controls.** On each of the three row sets, a zero update and the base model itself read answer gain and numerator
exactly 0. Every full-strength read equals E328's value for the same task and seed to all digits, a difference of
exactly 0, so the same rows and the same update give the same number on this path.

**Run 1.** One RunPod H100, $7.20 at $4.41 an hour, all on VPD decomposition p-9707d6ff.

**Run 2.** One RunPod H100 SXM, $3.62 at $4.41 an hour, about 49 minutes of pod, on the same decomposition. It
re-read the same fine-tunes on repacked rows, as described in its own section below.

## The share at each gain, per task and seed

Every value below is one fine-tune seed at its last save, on VPD p-9707d6ff. Gain is in whole-answer log-odds.
The last column is the share at gain about one minus the share at full gain, for the same seed.

| task | seed | full, $c = 1$: gain, share | about 10: $c$, gain, share | about 1: $c$, gain, share | share, about 1 minus full |
|---|---|---|---|---|---|
| known facts | 0 | 15.92, 0.597 | 0.139, 9.92, 0.654 | 0.00875, 1.012, 0.744 | +0.147 |
| known facts | 1 | 16.02, 0.576 | 0.139, 10.08, 0.653 | 0.00937, 1.063, 0.723 | +0.147 |
| known facts | 2 | 15.96, 0.583 | 0.139, 9.99, 0.656 | 0.00937, 1.063, 0.727 | +0.144 |
| copy | 0 | 10.50, 0.637 | 0.187, 9.65, 0.646 | 0.00937, 1.073, 0.579 | -0.059 |
| copy | 1 | 10.86, 0.611 | 0.187, 9.63, 0.648 | 0.0100, 1.084, 0.586 | -0.025 |
| copy | 2 | 10.58, 0.622 | 0.285, 10.20, 0.635 | 0.00937, 1.065, 0.589 | -0.033 |

**Reading it.** On known facts, seeds 0 to 2, the share rises as the fine-tune is scaled down. It goes from 0.58
to 0.60 at full gain, about 16, to 0.65 to 0.66 at gain about 10 and 0.72 to 0.74 at gain about 1. Per seed the
step from full gain to gain one is +0.147, +0.147 and +0.144.

On copy, seeds 0 to 2, the share reads 0.61 to 0.64 at full gain, 10.5 to 10.9, and 0.58 to 0.59 at gain about 1.
Per seed the step from full gain to gain one is -0.059, -0.025 and -0.033. Copy's own full gain is close to 10, so
its "about 10" read, at gains 9.6 to 10.2 against full gains 10.5 to 10.9, is barely a step down and not a separate
gain level. There it reads 0.63 to 0.65, so only the gain-one read is a real change of scale on copy.

The gain-one read is close to the share's small-update limit. As $c$ shrinks to 0, both answer changes shrink with
it and their ratio settles to a fixed value set by the update's direction alone. The full-gain read adds the
non-linear part on top. This limit is argued from the formula and was not computed.

## The same fine-tunes beside random updates at gain about one

E330 drew three random updates per task: Gaussian noise at the per-matrix norms of one fine-tune seed, with draw
seeds 228001 to 228003, its sign chosen and one scale factor searched until the gain sat in the band 1.003 to 1.097.
The same rows and share as above. "Size" is the draw's norm over the fine-tune seed's norm it was drawn from.

| task | draw | size, times the fine-tune's | gain | KL per position | share |
|---|---|---|---|---|---|
| known facts | 0 | 12.1 | 1.010 | 0.0132 | 0.687 |
| known facts | 1 | 18.7 | 1.055 | 0.0359 | 0.515 |
| known facts | 2 | 23.3 | 1.037 | 0.0662 | 1.031 |
| known facts | 2, rows repacked | 23.3 | 1.054 | 0.0612 | 1.211 |
| copy | 0 | 8.8 | 1.083 | 0.0270 | 0.719 |
| copy | 1 | 17.4 | 1.024 | 0.1123 | 0.630 |
| copy | 2 | 13.4 | 1.065 | 0.0680 | 0.666 |

**Reading it.** On known facts, the fine-tunes at gain about one, seeds 0 to 2, read 0.72 to 0.74. The random
draws 0 to 2 at gain about one read 0.687, 0.515 and 1.031, so the fine-tunes sit inside the draws' range. On copy,
the fine-tunes at gain about one, seeds 0 to 2, read 0.58 to 0.59 on these rows and the draws 0 to 2 read 0.630,
0.666 and 0.719, so on this packing the fine-tunes read below every draw. On repacked rows, run 2 below, copy seeds 0
and 1 at gain about one read 0.628 and 0.630, level with the lowest draw rather than below it. The draws were read on
the original packing only, apart from known draw 2. On neither task is there any sign that a fine-tune reads above a
random update of the same answer gain.

**The full-gain comparison mixes in the gain.** Appendix H, as updated by P2.90, sets the fine-tunes at their own full
gain beside random draws at gain about one. On known facts that comparison sets 0.58 to 0.60, at gain about 16,
against 0.51 to 1.03 at gain about 1. Scaling the same fine-tunes to gain one moves them by +0.144 to +0.147, so
about 0.15 of any known-facts gap in that comparison comes from the difference in gain, not from the update. On copy
the same step is -0.025 to -0.059 on the original rows and -0.006 and +0.020 on repacked rows. On cipher, invented facts and both maths tasks the random draws were also read at
gain about one, against fine-tunes at gains 16 to 160. E332 did not read those tasks, so their gaps are stated with
both gains and are not gain-matched.

## The random known draw is unstable under repacking

We read E330's known-facts random draw 2 again with the same draw seed, the same size of 46.02 and the same scale
factor of 23.34, so the update is the same tensor at the same scale. Only the rows changed. The same 600 trained
known-facts items were packed into rows in a new order and cut at new places, giving 1,081 whole answers against
the original 1,066.

The share moved from 1.031 to 1.211, a change of 0.18. The gain barely moved, 1.037 to 1.054, a change of 0.017,
while the numerator rose from 1.070 to 1.276, by 0.21. Same rows and same update reproduce exactly on this path: the
full-strength reads of known and copy, seeds 0 to 2, equal E328's to every digit, read on a separate pod. So the 0.18 is row packing, not read noise.

One draw's share therefore carries a sensitivity to which rows are read of about 0.2 on one repack. That is larger
than the copy fine-tunes' steps from full gain to gain one, -0.025 to -0.059, and close in size to the known-facts
step, +0.144 to +0.147. Run 2, next, reads whether the fine-tunes' shares are as sensitive to repacking.

## Run 2 (repacked rows)

Run 2 re-read the known and copy fine-tunes, seeds 0 to 2, on each task's rows repacked with packing seed 1: the
same trained items packed into rows in a new order and cut at new places. That gives 1,081 whole answers on known
facts against the original 1,066, and 1,724 on copy against 1,732. Each update was read at $c = 1$ and at the exact
$c$ run 1 used for that seed's gain-one read, with no new search. So the update is held fixed and only the rows
change.

Because $c$ was not searched again, the gain-one reads land wherever the new rows put them. On known facts they read
1.017 to 1.068, inside the band. Copy seed 1 read 1.015, inside it. Copy seed 0 read 1.0025, just under the band's
lower edge of 1.003 but above the share's floor, so its share is read. Copy seed 2 read 0.995 and gave no reading,
because its gain is under the floor of 1 that the share requires.

**Controls.** A zero update and the base model read answer gain and numerator exactly 0 on both repacked row sets.
Known draw 2 read again on the repacked known rows gave 1.211, equal to run 1's repeat to every digit. So these are
the rows on which that one draw's share changed by 0.18, and the read reproduces on them.

| task | seed | read, $c$ | original rows: gain, share | repacked rows: gain, share | share, repacked minus original |
|---|---|---|---|---|---|
| known facts | 0 | full, 1 | 15.92, 0.597 | 15.73, 0.620 | +0.023 |
| known facts | 0 | about 1, 0.00875 | 1.012, 0.744 | 1.017, 0.760 | +0.016 |
| known facts | 1 | full, 1 | 16.02, 0.576 | 15.85, 0.599 | +0.022 |
| known facts | 1 | about 1, 0.00937 | 1.063, 0.723 | 1.067, 0.743 | +0.020 |
| known facts | 2 | full, 1 | 15.96, 0.583 | 15.80, 0.603 | +0.020 |
| known facts | 2 | about 1, 0.00937 | 1.063, 0.727 | 1.068, 0.749 | +0.022 |
| copy | 0 | full, 1 | 10.50, 0.637 | 10.19, 0.634 | -0.003 |
| copy | 0 | about 1, 0.00937 | 1.073, 0.579 | 1.0025, 0.628 | +0.049 |
| copy | 1 | full, 1 | 10.86, 0.611 | 10.57, 0.610 | -0.001 |
| copy | 1 | about 1, 0.0100 | 1.084, 0.586 | 1.015, 0.630 | +0.044 |
| copy | 2 | full, 1 | 10.58, 0.622 | 10.31, 0.627 | +0.004 |
| copy | 2 | about 1, 0.00937 | 1.065, 0.589 | 0.995, no reading | none |

The step from full gain to gain about one, per seed, on each packing:

| task | seed | original rows | repacked rows |
|---|---|---|---|
| known facts | 0 | +0.147 | +0.140 |
| known facts | 1 | +0.147 | +0.145 |
| known facts | 2 | +0.144 | +0.146 |
| copy | 0 | -0.059 | -0.006 |
| copy | 1 | -0.025 | +0.020 |
| copy | 2 | -0.033 | no reading |

**Reading it.** Packing moves the fine-tunes' shares by a few hundredths (on copy's gain-one reads, together with the gain shift the repack caused at the same scale), far less than it moved the random draw. On
known facts, seeds 0 to 2, every share rose by 0.016 to 0.023 on the repacked rows, at full gain near 16 and at gain
about one alike. The step from full gain to gain one is therefore +0.140 to +0.146 on repacked rows against +0.144 to
+0.147 on the original rows, so known's rise of about 0.15 holds on both packings.

On copy, seeds 0 to 2, the full-gain shares moved by 0.004 at most, at gains 10.50 to 10.86 on the original rows and
10.19 to 10.57 on repacked rows. The gain-one shares of seeds 0 and 1 rose by 0.049 and 0.044, at gains 1.073 and
1.084 on the original rows against 1.0025 and 1.015 on repacked rows. Copy's step from full gain to gain one is
thus -0.059 and -0.025 on the original rows but -0.006 and +0.020 on repacked rows. Run 1's small decrease on copy
is within what one repack moves it, and copy's share shows no clear change with gain.

The largest move of any fine-tune share with packing is 0.049, on copy seed 0 at gain about one. Known draw 2's share
changed by 0.18 on one repack, at gain 1.037 on the original rows and 1.054 on the repacked ones. So the
sensitivity to packing seen in run 1 belongs to that large random update far more than to the fine-tunes.

**Beside the random draws.** The draws were read on the original packing only, apart from known draw 2. On known
facts, seeds 0 to 2 at gain about one read 0.74 to 0.76 on repacked rows, at gains 1.017 to 1.068. Draws 0 to 2 read
0.51 to 1.03 at gains 1.010 to 1.055, so the fine-tunes sit inside the draws' range on both packings. On copy, seeds
0 and 1 at gains 1.0025 and 1.015 read 0.628 and 0.630 on repacked rows. Draws 0 to 2 read 0.630 to 0.719 at gains
1.024 to 1.083. So on copy the fine-tunes read below the draws on the original rows and level with the lowest draw on
repacked rows, and never above them on either task.

## What we learned

The Qwen re-gating share on VPD depends on how far an update moves the answer, not only on the update's direction.
Scaling a known-facts fine-tune down to gain one raises its share by about 0.15 on both packings. On copy the change
is a few hundredths either way, depending on the packing. Once the gain is matched, the known-facts and copy
fine-tunes do not read above the random updates on either packing. Repacking the rows moves a fine-tune's share by at
most 0.05, while one random draw's share moved by 0.18. So the spread across draws, and their sensitivity to which
rows are read, matter more to that comparison than the fine-tunes' own row sensitivity.

## How compelling is the evidence

- Two tasks, three fine-tune seeds each, one decomposition, two packings of each task's rows for the fine-tunes.
- The read floor for the same rows and update is exactly 0 on this path. The row-set sensitivity is measured on one
  repack: at most 0.05 for the fine-tunes, 0.18 for the one random draw read again. The other random draws were
  read on one packing only.
- Copy seed 2 has no gain-one reading on repacked rows, and copy seed 0's repacked gain-one read sits just under
  the gain band.
- Gain is matched; norm and the KL side effect are not.
- The share is a ratio of two non-linear functions of $c$, so it is not expected to stay constant under scaling.
  The read has the power to show a change: $c$ near 0.009 takes the known-facts gain from 16 to 1.
- This run is descriptive: values are reported per gain with no bar.

## Surprises

The repeat of known draw 2 moved by 0.18 when only the row packing changed, more than the gain steps on copy. On
copy, seeds 0 and 1, repacking moved the gain-one shares by 0.044 and 0.049 while the full-gain shares moved by
0.004 at most, enough to take copy's step from full gain to gain one from a small decrease to about zero.

## What next

- Repacked reads of the other random draws: if they move by about 0.2 as known draw 2 did, a single draw cannot
  serve as a reference on these tasks; if they move by a few hundredths, draw 2 is the outlier.
- Gain-matched fine-tune reads on cipher, invented facts and the maths tasks would say whether their gaps to the
  random draws survive matching the gain.

## Departures from the literature

None. The share, the random draws and the gain search are this project's own, as in P2.90 and E330.

## Not run

- The random draws other than known draw 2 on repacked rows.
- A new gain search on the repacked rows, which would give copy seed 2 a gain-one reading.
- The figure. The plot script `experiments/share_unified/plot_gain.py` is written and passes ruff but was not run:
  the CPU node was down, and running it would have needed a pod. It plots share against gain, log scale, per task,
  with the three seeds joined and the random draws and the repeat beside them.
- Gain-matched reads on cipher, invented facts, maths and shuffled maths.
- The neuron-basis decomposition p-015e0710.
- The share's small-update limit, computed directly as a ratio of directional derivatives.
- Per-answer cells, which would let row-set sensitivity be estimated without a GPU; none were saved.
- A random update matched in both gain and norm, or in gain and KL.

## Provenance

- Code `experiments/share_unified/qwen_gain.py`; filed settings `experiments/share_unified/results/predictions_gain.json`.
- Results `experiments/share_unified/results/qwen_gain.jsonl`, commit 7caa976c; CSV
  `experiments/share_unified/results/qwen_share_vs_gain.csv`, built by `experiments/share_unified/build_csv.sh`,
  with a packing column, seed0 for run 1 and seed1 for run 2.
- Run 2: code `experiments/share_unified/qwen_repack.py`, filed settings
  `experiments/share_unified/results/predictions_repack.json`, results
  `experiments/share_unified/results/qwen_repack.jsonl`, sha256 815a27ec, commit bc09bb4d; logs `run_repack.log` and
  `run_repack_dry.log` beside it.
- E330's random draws `experiments/share_unified/results/qwen_random_vpd0.jsonl` and `qwen_random_vpd1.jsonl`.
- Review `reports/reviews/2026-11-27_review_e332-share-vs-gain.md`.
- Cost ledger: run 1 one RunPod H100 at $4.41 an hour, $7.20 in all; run 2 one RunPod H100 SXM at $4.41 an hour,
  $3.62 for about 49 minutes; neither is in the spend history, so both are stated here.
