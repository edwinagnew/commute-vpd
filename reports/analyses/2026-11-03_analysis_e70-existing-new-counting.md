# Does removing the old count hurt the reusing task more than the new one?

E70 and its Amendments 1 and 2, the last, benchmark line, 2026-11-03. Provisional reading, no independent check. Plan `reports/plans/2026-11-03_plan_e70-existing-new-counting.md`. Rules `experiments/pcfg_anchors/results/predictions_e70.json`, `predictions_e70_a1.json`, `predictions_e70_a2.json`. Results `results/e70.json`, commit a6103fc; `results/e70_a1.json`, commit afc1191; `results/e70_a2.json`, commit 71d3975. Notebook `notebooks/E70_existing_new_counting.ipynb`.

We removed single directions from a small counting model and from two fine-tunes of it, one built to reuse the old count and one to need a new one.
At blocks 3 to 5 the direction the count moves costs the reusing task more than the new task, while the largest direction with no covariance costs nothing, so the last filed branch is "count-specific at this size", and the earlier "cannot serve" stands.
For `GOALS.md`'s question of new against strengthened capabilities, this supports a causal role for the count's direction there, on one pair at an unmatched size, and the new task's loss stays unexplained.

## The answer

E70's two amendments end on two filed verdicts. "Cannot serve" stands, because the new task loses more at block 6. The last amendment reads "count-specific at this size". At blocks 3 to 5 the direction the count moves costs the pretrained model heavily and the reusing task more than the new one, while the largest direction with no covariance costs nothing. That supports a causal role for the count's direction there, on one pair, unmatched in size. The new task's loss stays unexplained.

**How we got here.** The first reading removed the regression probe's one direction, which did nothing, and six softmax directions, which did something. The first amendment removed the direction where the count moves the residual, and it explained why the first direction did nothing, but that direction holds about half the variance. The second amendment asks whether size alone explains its effect. Asks 15 and 17, whether to try each, were taken by default after two hours without an answer, so the user did not choose them. E70 ends here: a third amendment would be a new plan, and none is proposed.

## What we were trying to test

`GOALS.md` asks whether post-training teaches new capabilities or strengthens existing ones. `decisions.md` defines that without any decomposition, after Jain, Kirk, Lubana et al. 2024. A model possesses a capability if a linear readout trained on pretraining data recovers it. A fine-tune reuses it if the new map is some function of that reading. And reuse needs a causal claim: taking the readout's direction away must break the fine-tuned behaviour. That last clause had never been read on a pair whose answer is known by construction.

**The model.** LOW is a small transformer of 6 blocks and width 192, pretrained to count the letter a when asked `C a 40`.

**The pair.** Every state is LOW fine-tuned on `C b 40` rows of natural text.

- **Task P** answers the count of a plus one. It reuses the old count, by construction.
- **Task N** answers the count of b. It needs a new count, by construction.

Both tasks ran at two learning rates and five seeds, saved at 1,000 and 10,000 steps, the same grid as an earlier stage. Those states were lost with an old clone, so LOW and all 20 runs were retrained from the same code and seeds. The filed rule reads the 20 states at 10,000 steps.

**Two probes on LOW,** each trained per block on `C a 40` rows, read at the `=` position, with no possession threshold.

- **The ridge probe** regresses the count on the residual and gives one direction, its unit weight vector. This is the task as asked, and the filed headline.
- **The softmax probe** predicts the count as a class. The classes it sees are the counts 10 to 16, so its centred columns span six directions out of 192. It is reported beside the ridge probe, to ask whether one direction misses a count held in several.

**The removal.** With $h$ the residual after block $b$ at `=` and $U$ an orthonormal basis of the probe's directions, $h$ becomes $h - U U^\top h$ at that one position. Later blocks still read the text.

**The score.** Let $D_{\text{state}}$ be the state's mean drop in the log-odds of its own answer under the removal, $D_{\text{LOW}}$ be LOW's drop for the same answer on the same rows, and $G$ be the state's mean log-odds gain over LOW. The share of the gain lost is $(D_{\text{state}} - D_{\text{LOW}}) / G$. A share of 0 means the state lost no more than LOW did, and 1 means the whole gain went. The raw share is $D_{\text{state}} / G$.

**The controls,** each read in the same job, and what each rules out:

- LOW's count accuracy against the old run, to rule out a different model under the same name.
- LOW's own log-odds drop under the removal, on `C a 40` rows, to rule out a removal that does nothing. If it costs LOW nothing, a null on the tasks means nothing.
- 20 random directions per block, or 20 random six-direction subspaces, on LOW and on every state. The 95th percentile of their size sets the band at each block, so any removal of that size has to beat it.

**The rule.** At a block, the smallest P share minus the largest N share must exceed that block's band. All ten states of a side must clear all ten of the other.

**The amendment.** The first reading, below, left one direction that reads the count perfectly and does nothing when removed. Ask 15, whether to try a direction chosen another way, was taken by default at 15:28 on 2026-11-03. Two directions are compared from here on.

- **The covariance direction** is the unit covariance of the residual with the count of a, per block, on the rows the ridge probe was fit on. It is where the count moves the residual.
- **The regression direction** is the ridge probe's unit weights, the direction the first reading removed.

The removal, the rows, the random draws and the score are the first reading's, and nothing is retrained. LOW is read first, and the tasks only if LOW loses log-odds. Three things are new, each to rule something out:

- **Accuracy after removal,** because a large log-odds drop can leave the answer in place.
- **A ridge refit on fresh rows after each removal,** to ask whether the count can still be read at all. It rules out a removal that does not do what it claims.
- **The share of the residual's variance along each direction,** to test why the two directions differ.

**The filed tree.** A block is read only if LOW loses log-odds there above the band. Then the first match wins: a reversal at any read block gives "cannot serve"; otherwise the rule holding at a read block from 1 to 5 gives "separates". So a single reversal decides, whatever else holds.

**The second amendment, the last.** The covariance direction holds 0.30 to 0.78 of the residual's variance, and the random directions that set the band hold under one percent, so a loss above the band could be size alone. Ask 17, whether to control for that, was taken by default at 18:08 on 2026-11-03, so the user did not choose this either. Only the direction changes. LOW, its rows, the 20 states at 10,000 steps, the random draws and the score are Amendment 1's, and nothing is retrained.

- **The no-covariance direction** is the top principal component of LOW's residual once the covariance direction is removed, per block, on the same 20,000 fit rows. It is at right angles to the covariance direction, so its covariance with the scalar count on those rows is zero by algebra.
- **It cannot match on size at blocks 1 to 5.** A direction at right angles to the covariance direction holds at most what that direction leaves: about 0.22, 0.49, 0.44, 0.44 and 0.47. This is the largest single direction available, not a size-matched one.
- **The class share** is new, and it reads the share of a direction's variance lying between the classes of a count, on LOW's 2,000 held rows. It is the correlation ratio squared: 0 means every class has the same mean along the direction, and 1 means all its variance lies between classes. It rules out a false null, because zero covariance with the scalar count says nothing about a curve through the classes, and LOW's head is a softmax over those classes.

**The filed tree for this amendment.** Each of blocks 1, 3, 4 and 5, where Amendment 1's rule held, gets one label under the no-covariance direction.

- **Size:** the rule holds again, with under half of the direction's variance between count classes.
- **Carrier:** the rule holds again, with at least half between count classes.
- **Specific:** the rule does not hold, and P's mean share under the covariance direction exceeds P's mean share under this one by more than the band.

The first match wins among the branches. A size block and a specific block give "mixed"; a size block alone gives "size explains"; a specific block without a size block gives "count-specific at this size"; a carrier block alone gives "carries the classes". Block 6 is read apart, and nothing here can change Amendment 1's filed "cannot serve".

## What we expected, and why

We expected the removal to work on LOW, at least at the last block, since a readout that reads the count should be using what the model uses.

We expected the rule to hold. P's answer is a function of the old count, so removing the count should cost P a lot: over half its gain at block 6. N's answer does not depend on the count of a, so N should sit near the band at blocks 1 to 5, and might rise at block 6, where every answer shares the output head.

We expected the function test to split the pair cleanly. A table from LOW's reading to the state's answer should score near the probe's accuracy on P and near its floor on N.

The plan named one blind spot in advance: the count of a and the count of b are correlated on these rows, so a direction shared between them would cost N as well.

**For the amendment,** we expected less damage than we saw. LOW would lose about 0.3 to 5 log-odds at some of blocks 3 to 6, with low confidence, and keep accuracy at or above about 0.7, because the softmax head can still separate classes in other directions. The refit would read the count near the 0.3575 floor after the covariance direction, which is close to algebra, and above 0.9 after the regression direction from block 3.

Task P would lose more than the band and less than under the six softmax directions at most blocks, most at blocks 4 and 5. Task N would stay under P at blocks 4 and 5 and above the band at blocks 4 to 6, up to about its six-direction values of 0.20, 0.34 and 0.49. The weak lean was that the rule would hold at block 4 or 5 and not at block 6. "Cannot serve" was named as a live outcome because the two counts correlate.

**For the second amendment,** we leaned weakly to "count-specific at this size" or "mixed", since the no-covariance direction holds less variance than the covariance direction at blocks 1 to 5. "Size explains" and "carries the classes" were named as live. We expected the direction to hold 0.1 to 0.3 of the variance, with low confidence. We expected it to cost LOW more than the band at some blocks and less than the covariance direction at blocks 1 to 5, and to leave LOW's accuracy at or above about 0.7. We expected P's share to stay below Amendment 1's at blocks 1 to 5 and above the band at most blocks. We filed no lean for block 6, for N against P, or for the direction's class share, since the second component of an ordered set of classes is often curved in the label.

## What we observed

The first four questions are the first reading, on the two probes. The three after them are the first amendment. The five after those are the second amendment, and with the first they are what the answer above rests on.

### Does LOW hold the count of a?

LOW's count of a read 0.99975, equal to the old run, and its count of b 0.2215, also equal. All 10K states read task accuracy 1.0, at most 0.0005 from the earlier stage. At 1K they differ by at most 0.015, and the five N runs at 1e-5 read 0.483 to 0.497, as they did before: not yet learned, as filed. The correlation of count(a) and count(b) on `C b 40` rows is -0.50.

| block | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| ridge accuracy, `C a 40` (floor 0.3575) | 0.380 | 0.9975 | 1 | 1 | 1 | 1 |
| ridge accuracy, `C b 40` (floor 0.348) | 0.371 | 0.9985 | 1 | 1 | 1 | 1 |
| softmax accuracy, `C a 40` | 0.3555 | 0.919 | 0.9715 | 0.995 | 0.9975 | 1 |
| shuffled labels | 0.3305 | 0.3305 | 0.3345 | 0.3385 | 0.3535 | 0.356 |
| earlier run, same seed | 0.342 | 0.871 | 0.9155 | 0.9845 | 0.9985 | 1 |
| softmax accuracy, `C b 40` (floor 0.348) | 0.3665 | 0.910 | 0.9635 | 0.9955 | 0.9985 | 1 |

**Reading it.** Yes, from block 2 on, and the first two controls pass. LOW retrained to its old accuracy exactly, and every 10K state performs its own task perfectly. The table shows both probes on held-out `C a 40` rows. The ridge probe always beats its floor of 0.36, the rate of always guessing the commonest count, and it reads almost perfectly from block 2. The softmax probe beats its shuffled-label twin by a wide margin from block 2 and matches the earlier run at blocks 5 and 6, reading a little higher at blocks 2 and 3. At block 1 neither probe reads much.

![LOW's drop](../../experiments/pcfg_anchors/figures/e70_low.png)

### Does removing the probe cost LOW its own count?

**Reading it.** The ridge direction does not, at any block, and the softmax directions do, at every block. The figure's left panel plots LOW's log-odds drop against 20 random single directions. The ridge removal stays inside or under the grey band everywhere, and at blocks 4 and 5 it nudges the log-odds up, not down. Single random directions moved LOW by more than the probe's direction did. The right panel, on a scale about twenty times larger, has the softmax removal above its band at all six blocks, at most 9 log-odds against a band of about 0.5.

So one direction that reads the count perfectly can be taken away with no effect on the model that holds the count. Why stayed open until the amendment, below. Whatever follows on the ridge probe has no working control under it.

| block | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| ridge: LOW's drop | 0.042 | 0.027 | -0.035 | -0.104 | -0.122 | -0.020 |
| ridge: band | 0.048 | 0.132 | 0.215 | 0.122 | 0.117 | 0.077 |
| softmax: LOW's drop | 1.07 | 1.42 | 4.22 | 7.67 | 9.02 | 6.38 |
| softmax: band | 0.17 | 0.20 | 0.28 | 0.44 | 0.47 | 0.36 |

The filed test for an effect is one-sided: a removal that raises the log-odds loses nothing. At block 5 the ridge removal raises them by 0.122, just past the mirror of its band at 0.117, and this still counts as no effect.

![Ridge shares](../../experiments/pcfg_anchors/figures/e70_ridge.png)

### Does removing the one ridge direction cost task P more than task N?

**Reading it.** No: it costs neither task anything. The filed verdict is "does not separate", and this misses our lean that the rule would hold. Every share at every block lies within 0.01 of zero, on the same axes as the next figure. This cannot be read as a negative about the definition, because the removal does nothing on LOW either. The filed rule also returned a reversal at block 1, with N above P by 0.005 against a band of 0.004. That is half a percent of the gain, at the block where this probe reads barely above its floor, on a removal with no effect on LOW.

| block | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| P mean, 10K | -0.007 | 0.004 | 0.005 | 0.003 | 0.003 | -0.003 |
| N mean, 10K | 0.002 | 0.001 | 0.000 | 0.001 | -0.002 | -0.001 |
| P raw mean | -0.001 | 0.003 | 0.001 | -0.003 | -0.005 | -0.001 |
| N raw mean | 0.002 | 0.001 | 0.000 | 0.001 | -0.006 | 0.000 |
| band | 0.0042 | 0.024 | 0.012 | 0.019 | 0.011 | 0.009 |
| smallest P minus largest N | -0.012 | 0.001 | 0.003 | -0.002 | -0.001 | -0.003 |
| smallest N minus largest P | 0.0050 | -0.005 | -0.006 | -0.004 | -0.012 | 0.001 |
| rule | reversal | neither | neither | neither | neither | neither |

The largest size of any ridge share, at any block or either save, is 0.0098.

![Softmax shares](../../experiments/pcfg_anchors/figures/e70_softmax.png)

### Does removing the six softmax directions cost task P more than task N?

**Reading it.** Yes at blocks 2 to 5, by the filed rule, but it is a difference of degree and not the clean split we predicted. In the figure the whole P range clears the whole N range by more than the band at those four blocks, most widely at block 5, where P loses 0.72 of its gain and N loses 0.34. At blocks 1 and 6 the rule does not hold. P passes the half-gain mark at block 6, as leaned, with 0.56. N misses its prediction of staying near the band up to block 5. It reads 0.20 at block 4 and 0.34 at block 5, against bands of 0.045 and 0.063.

This removes six directions, not the one the task named, so it does not by itself answer the task as asked.

**What subtracting LOW's drop does.** The scored share subtracts LOW's drop, and LOW's drop on these rows is negative. Removing its count directions raises LOW's log-odds of P's answer by about 5 at blocks 4 and 5, and of N's answer by about 3. Subtracting a negative number adds to the share, as the gap between the solid and dashed lines shows: about 0.19 for P, and 0.13 to 0.16 for N. In the raw share the gap between the tasks survives at blocks 4 and 5: the smallest P is 0.28 and 0.50, and the largest N is 0.08 and 0.22. At blocks 2 and 3 the raw gap is 0.01 and 0.04, under the scored band, so those two blocks lean on the subtraction. At block 6 the raw shares are equal, at 0.32.

**A candidate for N's rise, not a cause.** The count of a and the count of b correlate at -0.50 on these rows, the blind spot the plan named. A direction shared by the two counts would cost N as well. This run did not test that.

**At 1,000 steps** the shares run higher than at 10,000 on both tasks, by up to about 0.14 on N and 0.12 on P. Five of the ten N states there have not yet learned the task, at about 0.49 accuracy, so the filed rule leaves 1K out. The gain is also smaller there, about 21 log-odds on P and 11 on N against 26 and 22, while LOW's subtracted term stays fixed.

| block | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| P mean, 10K | 0.061 | 0.125 | 0.238 | 0.553 | 0.720 | 0.564 |
| P min to max | 0.044-0.077 | 0.097-0.154 | 0.186-0.287 | 0.454-0.620 | 0.670-0.756 | 0.535-0.591 |
| N mean, 10K | 0.020 | 0.042 | 0.102 | 0.195 | 0.339 | 0.494 |
| N min to max | 0.010-0.031 | 0.040-0.045 | 0.076-0.130 | 0.158-0.230 | 0.248-0.408 | 0.456-0.527 |
| band | 0.019 | 0.041 | 0.041 | 0.045 | 0.063 | 0.035 |
| smallest P minus largest N | 0.012 | 0.052 | 0.056 | 0.224 | 0.262 | 0.007 |
| rule | neither | holds | holds | holds | holds | neither |
| P raw mean | 0.055 | 0.043 | 0.117 | 0.359 | 0.527 | 0.321 |
| P raw min to max | 0.038-0.071 | 0.023-0.064 | 0.078-0.154 | 0.279-0.408 | 0.495-0.545 | 0.314-0.326 |
| N raw mean | 0.017 | 0.004 | 0.027 | 0.069 | 0.179 | 0.316 |
| N raw min to max | 0.006-0.028 | -0.004-0.013 | 0.013-0.043 | 0.052-0.083 | 0.113-0.221 | 0.307-0.320 |
| LOW's drop for P's answer | -0.155 | -2.093 | -3.073 | -4.935 | -4.930 | -6.174 |
| LOW's drop for N's answer | -0.073 | -0.819 | -1.591 | -2.689 | -3.415 | -3.793 |
| P's own drop, mean | 1.394 | 1.065 | 2.930 | 9.135 | 13.511 | 8.259 |
| N's own drop, mean | 0.398 | 0.107 | 0.552 | 1.470 | 3.775 | 6.916 |
| P mean, 1K | 0.063 | 0.154 | 0.357 | 0.669 | 0.793 | 0.618 |
| N mean, 1K | 0.021 | 0.089 | 0.222 | 0.330 | 0.475 | 0.576 |
| share of gain in nats, P, unscored | 0.000 | 0.000 | 0.013 | 0.155 | 0.308 | 0.001 |
| share of gain in nats, N, unscored | 0.000 | 0.000 | 0.001 | 0.002 | 0.013 | 0.002 |

The mean log-odds gain over LOW is 25.7 on P and 21.9 on N at 10K, and 21.1 and 10.8 at 1K. `readout.shares` takes the state's drop minus LOW's drop, over the gain, as the scored share. The raw share is the state's drop over the gain, so the two differ by exactly LOW's drop over the gain. LOW's drop is one number per task.

### Is each task's answer a function of LOW's reading?

**Reading it.** Yes for P, no for N, as predicted, on all ten states of each side at 10K. The table is a lookup from LOW's rounded ridge reading to the state's answer, fitted on half the rows and scored on the other half. On P it follows the probe's accuracy and reaches 1 from block 3. On N it stays at 0.39 from block 2, against a floor of 0.36. The reading comes from the ridge probe, so this direction reads the count well enough to give P's answer exactly, while removing it does nothing. This test reads answers alone, and the answers are fixed by each task's construction, so it checks the pair more than it checks the removal.

| block | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| lookup on P, 10K, all ten states (floor 0.353) | 0.367 | 0.997 | 1 | 1 | 1 | 1 |
| lookup on N, 10K, all ten states (floor 0.357) | 0.343 | 0.386 | 0.387 | 0.387 | 0.387 | 0.387 |

At 1K, P reads the same. The five N runs at 1e-5, not yet learned, read 0.52 to 0.57 at block 6, against floors of 0.43 to 0.54.

### Why did the regression direction do nothing?

**Reading it.** Because the count is still readable without it. After removing the regression direction, a fresh ridge refit reads the count at 1.0 from block 3 and at 0.993 at block 2, against 0.9975 with nothing removed. After removing the covariance direction, the same refit reads 0.3575 to 0.3705, against a floor of 0.3575. The number moved, so this answers the first reading's puzzle. The regression direction holds 0.00006 to 0.03 of the residual's variance, the covariance direction 0.30 to 0.78, and the cosine between them is at most 0.32.

The fall to the floor is close to algebra, since zero covariance leaves a linear reader of the count nothing, so it checks the removal and is not a discovery. Variance alone does not explain the regression direction's silence either. At blocks 4 to 6 it holds three to seven times what a random direction holds, and it still costs LOW nothing.

| block | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| refit, nothing removed (floor 0.3575) | 0.380 | 0.9975 | 1 | 1 | 1 | 1 |
| refit after the regression direction | 0.3735 | 0.993 | 1 | 1 | 1 | 1 |
| refit after the covariance direction | 0.3575 | 0.359 | 0.369 | 0.3655 | 0.3705 | 0.362 |
| variance held, covariance direction | 0.776 | 0.505 | 0.560 | 0.561 | 0.530 | 0.297 |
| variance held, regression direction | 0.00006 | 0.0017 | 0.0035 | 0.0160 | 0.0316 | 0.0255 |
| variance held, random, mean of 20 | 0.0045 | 0.0039 | 0.0067 | 0.0043 | 0.0045 | 0.0050 |
| cosine of the two directions | 0.012 | 0.062 | 0.083 | 0.171 | 0.247 | 0.320 |

### Does removing the covariance direction cost LOW its count?

![LOW under the covariance direction](../../experiments/pcfg_anchors/figures/e70_a1_low.png)

**Reading it.** Yes by the filed test, at all six blocks, and by much more than we filed. The figure plots LOW's log-odds drop per block. The covariance direction drops it 1.9 to 12, against bands of at most 0.22, while the regression direction stays on the zero line with the random draws. LOW's accuracy on its own count falls from 1.0 to 0.36 at block 5, which is the floor of always answering the commonest count. At blocks 1 to 5 this one direction costs LOW more log-odds than the six softmax directions did. The next section limits what this can be read as.

| block | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| LOW's drop, covariance direction | 1.93 | 4.21 | 8.11 | 10.94 | 11.96 | 1.88 |
| band | 0.048 | 0.132 | 0.215 | 0.122 | 0.117 | 0.077 |
| LOW's drop, regression direction | 0.042 | 0.027 | -0.035 | -0.104 | -0.122 | -0.020 |
| LOW's drop, six softmax directions | 1.07 | 1.42 | 4.22 | 7.67 | 9.02 | 6.38 |
| LOW's accuracy, covariance direction | 0.979 | 0.828 | 0.573 | 0.371 | 0.3605 | 0.9995 |

LOW's accuracy with nothing removed reads 1.0 on these rows, and its mean stays 1.0 under the random directions and under the regression direction.

### Can that loss be read as the count's causal role?

**Reading it.** Not from this amendment alone, and this limits the result as much as the verdict does. The band comes from random single directions, which hold about 0.004 to 0.007 of the residual's variance. The covariance direction holds 0.30 to 0.78, which is 60 to 170 times as much. So the band shows that a small random direction does little. It says nothing about what a direction this large would do if it had no tie to the count.

Removing half of a residual's variance could hurt for reasons unrelated to the count. The bands on the tasks come from the same random directions, so the same limit applies there. So "LOW loses log-odds above the band" and the gap between the tasks at blocks 3 to 5 cannot be read as the count's causal role on this amendment's evidence.

Neither the plan nor the filed list of what the measurement is blind to named this. The plan filed the variance share as a reading, used only to test why the two directions differ, and the limit was found on reading the result. It does not touch the filed verdict, which stands.

The control this calls for removes a direction of matching variance with no covariance with the count, and it has its own limit. Such a direction is at right angles to the covariance direction, so it holds at most what that direction leaves: 0.22, 0.49, 0.44, 0.44, 0.47 and 0.70 at blocks 1 to 6. An exact match is therefore impossible at blocks 1 to 5, and only the largest such direction can be removed there. The second amendment, below, does that.

### Does removing the covariance direction cost task P more than task N?

![Tasks P and N under the covariance direction](../../experiments/pcfg_anchors/figures/e70_a1_tasks.png)

**Reading it.** The filed verdict is "cannot serve", because of a reversal: at block 6, task N loses more than task P, by 0.081 against a band of 0.009. LOW has teeth at all six blocks, so all six are read, and the tree's first-match rule makes any reversal decide. The verdict stands. Blocks 1 and 3 to 5 go the other way by the filed gap rule, and block 2 is neither. The rule read over all blocks printed "separates" beside the branch, and the tree is what decides.

At blocks 3 to 5, P loses 0.427, 0.779 and 0.884 of its gain and N loses 0.158, 0.335 and 0.493. Answers say the same. P's accuracy falls to 0.707, 0.479 and 0.285, the last below the 0.35 of always answering the commonest count. N is not spared: it falls to 0.980, 0.921 and 0.754. The two counts correlate at -0.50 on these rows, so this direction could carry the count of b too. That is a candidate for N's loss that this run did not test, not its cause.

**What was read at block 6.** N's share is 0.174 and P's is 0.070. In nats, the mean share of the gain is under 0.0001 on both tasks, with 0.00016 the largest on any state, and accuracy is 1.0 on every state. The raw share, which does not subtract LOW's drop and has no band, reads P 0.071 and N 0.045. LOW's drop for N's answer there is -2.73, and for P's answer 0.04. These are what was read. The filed rule reads the scored share, so they leave the branch as it is.

| block | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| P mean, 10K | 0.133 | 0.160 | 0.427 | 0.779 | 0.884 | 0.070 |
| P min to max | 0.097-0.166 | 0.094-0.219 | 0.283-0.523 | 0.713-0.825 | 0.850-0.915 | 0.066-0.079 |
| N mean, 10K | 0.032 | 0.067 | 0.158 | 0.335 | 0.493 | 0.174 |
| N min to max | 0.024-0.041 | 0.041-0.090 | 0.096-0.215 | 0.269-0.379 | 0.402-0.550 | 0.160-0.187 |
| band | 0.0042 | 0.0241 | 0.0125 | 0.0191 | 0.0114 | 0.0087 |
| smallest P minus largest N | 0.0555 | 0.0036 | 0.0678 | 0.3336 | 0.3000 | -0.1215 |
| smallest N minus largest P | -0.1419 | -0.1779 | -0.4268 | -0.5566 | -0.5126 | 0.0811 |
| rule | holds | neither | holds | holds | holds | reversal |
| P accuracy after removal, mean | 0.991 | 0.942 | 0.707 | 0.479 | 0.285 | 1 |
| P accuracy, min to max | 0.980-0.9995 | 0.888-0.997 | 0.587-0.881 | 0.447-0.503 | 0.175-0.467 | 1-1 |
| N accuracy after removal, mean | 1 | 0.999 | 0.980 | 0.921 | 0.754 | 1 |
| N accuracy, min to max | 1-1 | 0.998-1 | 0.955-1 | 0.868-0.983 | 0.637-0.925 | 1-1 |
| P raw mean | 0.080 | 0.109 | 0.342 | 0.676 | 0.830 | 0.071 |
| N raw mean | 0.002 | 0.001 | 0.056 | 0.176 | 0.322 | 0.045 |
| P's own drop, mean | 1.99 | 2.69 | 8.59 | 17.32 | 21.35 | 1.84 |
| N's own drop, mean | 0.03 | -0.02 | 1.11 | 3.83 | 6.97 | 1.03 |
| LOW's drop for P's answer | -1.354 | -1.304 | -2.178 | -2.644 | -1.378 | 0.040 |
| LOW's drop for N's answer | -0.649 | -1.406 | -2.175 | -3.380 | -3.643 | -2.732 |
| share of gain in nats, P, unscored | 0.0020 | 0.0205 | 0.186 | 0.650 | 0.833 | 0.00003 |
| share of gain in nats, N, unscored | 0.000001 | 0.0002 | 0.005 | 0.021 | 0.071 | 0.00007 |

Every state reads accuracy 1.0 with nothing removed. The mean log-odds gain over LOW is 25.7 on P and 21.9 on N. Per-state values are in `results/e70_a1.json`.

### Does the no-covariance direction cost LOW what the covariance direction cost?

![LOW under the two directions](../../experiments/pcfg_anchors/figures/e70_a2_low.png)

**Reading it.** No at blocks 3 to 5, and at block 6 it costs LOW more. The figure plots LOW's log-odds drop per block under each direction, labelled with the variance each holds. At blocks 3 to 5 the covariance direction costs LOW 8 to 12 log-odds and takes its count down to 0.36 at block 5. The no-covariance direction, holding 0.09 to 0.13 of the variance, stays inside the band there and leaves accuracy at 1.0. At blocks 1 and 2 it sits just above the band, at 0.06 against 0.05 and 0.20 against 0.13. Block 6 is the exception, read below.

| block | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| LOW's drop, no-covariance direction | 0.057 | 0.200 | 0.061 | -0.012 | 0.033 | 3.275 |
| LOW's drop, covariance direction | 1.927 | 4.209 | 8.112 | 10.939 | 11.965 | 1.882 |
| band | 0.048 | 0.132 | 0.215 | 0.122 | 0.117 | 0.077 |
| above the band, no-covariance direction | yes | yes | no | no | no | yes |
| LOW's accuracy, no-covariance direction | 1 | 1 | 1 | 1 | 1 | 0.9715 |
| LOW's accuracy, covariance direction | 0.979 | 0.828 | 0.573 | 0.371 | 0.3605 | 0.9995 |
| variance held, no-covariance direction | 0.224 | 0.265 | 0.134 | 0.096 | 0.086 | 0.264 |
| variance held, covariance direction | 0.776 | 0.505 | 0.560 | 0.561 | 0.530 | 0.297 |
| variance held, random, mean of 20 | 0.0045 | 0.0039 | 0.0067 | 0.0043 | 0.0045 | 0.0050 |
| class share of count(a), no-covariance direction | 0.006 | 0.006 | 0.012 | 0.015 | 0.017 | 0.988 |
| class share of count(a), covariance direction | 0.071 | 0.884 | 0.934 | 0.978 | 0.980 | 0.990 |
| class share of count(a), random, mean of 20 | 0.044 | 0.362 | 0.443 | 0.370 | 0.383 | 0.777 |
| class share of count(b), no-covariance direction | 0.039 | 0.004 | 0.003 | 0.003 | 0.113 | 0.075 |
| class share of count(b), covariance direction | 0.028 | 0.292 | 0.262 | 0.259 | 0.256 | 0.210 |
| refit after the no-covariance direction (floor 0.3575) | 0.3815 | 0.9165 | 0.9975 | 1 | 1 | 0.9995 |
| refit, nothing removed | 0.38 | 0.9975 | 1 | 1 | 1 | 1 |

The apparatus checks all hold. The ridge's accuracy and floor equal the first reading's, and the random drops differ from it by 0. LOW's drop under the covariance direction, rerun from the saved direction, equals Amendment 1's at all six blocks. The no-covariance direction's covariance with the count is at most 1.1e-8 of the covariance direction's on the fit rows, and at most 0.016 on the held rows, which is sampling error. The refit stays above its midpoint at blocks 2 to 6, so the count stays linearly readable after this removal.

### Does the no-covariance direction reproduce the gap between P and N?

![Tasks P and N under the two directions](../../experiments/pcfg_anchors/figures/e70_a2_tasks.png)

**Reading it.** No. The filed branch is "count-specific at this size", with the label "specific" at blocks 1, 3, 4 and 5. In the figure's left panel the solid lines are Amendment 1's gap, and the dashed lines lie on zero up to block 5. The right panel closes in on those dashed lines against the band. At blocks 3 to 5, all 20 states sit inside the band, and the mean shares are within 0.006 of zero on both tasks. The rule reads "neither" at all six blocks, so this direction does not separate the tasks anywhere. The states keep their answers too: accuracy reads at least 0.9995 at blocks 1 to 5, and the share of the gain in nats stays under 0.0001.

**Size in proportion does not explain it.** Scaling P's loss under the covariance direction down by the ratio of the two variance shares predicts 0.10, 0.13 and 0.14 at blocks 3 to 5, against -0.005 to 0.001 read. That scaling assumes loss proportional to variance, which nothing here establishes, and the share per unit of variance below is descriptive and unscored.

**Size as a threshold is not excluded.** Removing half the variance might cost something that removing a tenth does not, whatever the direction. No single direction with no covariance can match the covariance direction's size at these blocks, so this run cannot test that.

| block | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| label under the filed tree | specific | not read | specific | specific | specific | read apart |
| rule under the no-covariance direction | neither | neither | neither | neither | neither | neither |
| rule under the covariance direction | holds | neither | holds | holds | holds | reversal |
| band | 0.0042 | 0.0241 | 0.0125 | 0.0191 | 0.0114 | 0.0087 |
| P mean, no-covariance direction | -0.0005 | 0.0318 | 0.0009 | -0.0051 | -0.0010 | 0.2082 |
| P min to max | -0.0014 to 0.0009 | 0.0215 to 0.0407 | -0.0027 to 0.0037 | -0.0067 to -0.0036 | -0.0020 to 0.0023 | 0.198 to 0.217 |
| N mean, no-covariance direction | 0.0046 | 0.0094 | 0.0009 | -0.0001 | 0.0048 | 0.2020 |
| N min to max | -0.0008 to 0.0095 | 0.0054 to 0.0135 | 0.0005 to 0.0014 | -0.0016 to 0.0015 | -0.0002 to 0.0112 | 0.187 to 0.218 |
| P mean, covariance direction | 0.133 | 0.160 | 0.427 | 0.779 | 0.884 | 0.070 |
| N mean, covariance direction | 0.032 | 0.067 | 0.158 | 0.335 | 0.493 | 0.174 |
| smallest P minus largest N | -0.0109 | 0.0080 | -0.0042 | -0.0083 | -0.0132 | -0.0195 |
| smallest N minus largest P | -0.0017 | -0.0352 | -0.0031 | 0.0020 | -0.0026 | -0.0301 |
| P under the covariance direction, scaled by variance | 0.038 | 0.084 | 0.103 | 0.134 | 0.143 | 0.062 |
| share per unit of variance, P, covariance direction | 0.172 | 0.317 | 0.763 | 1.389 | 1.669 | 0.234 |
| share per unit of variance, N, covariance direction | 0.041 | 0.132 | 0.282 | 0.597 | 0.931 | 0.584 |
| share per unit of variance, P, no-covariance direction | -0.002 | 0.120 | 0.007 | -0.053 | -0.011 | 0.788 |
| share per unit of variance, N, no-covariance direction | 0.020 | 0.036 | 0.006 | -0.001 | 0.056 | 0.765 |
| P raw mean, no-covariance direction | 0.0017 | 0.0177 | 0.0013 | -0.0038 | 0.0027 | 0.1012 |
| N raw mean, no-covariance direction | 0.0050 | 0.0050 | -0.0005 | -0.0004 | 0.0081 | 0.1578 |
| P's own drop, mean | 0.043 | 0.440 | 0.028 | -0.102 | 0.069 | 2.607 |
| N's own drop, mean | 0.100 | 0.102 | -0.010 | -0.004 | 0.190 | 3.435 |
| LOW's drop for P's answer | 0.055 | -0.359 | 0.011 | 0.033 | 0.093 | -2.730 |
| LOW's drop for N's answer | 0.010 | -0.094 | -0.029 | -0.006 | 0.071 | -0.942 |
| P accuracy after removal, mean | 1 | 1 | 1 | 1 | 1 | 0.998 |
| N accuracy after removal, mean | 1 | 1 | 1 | 1 | 0.9999 | 0.997 |
| share of gain in nats, P, largest state | 0.000006 | 0.000007 | below 0 | below 0 | below 0 | 0.0016 |
| share of gain in nats, N, largest state | 0.00002 | 0.000003 | 0.000001 | 0.000003 | 0.00006 | 0.0064 |

Every state reads accuracy 1.0 with nothing removed, and the random state band equals Amendment 1's. Per-state values are in `results/e70_a2.json`.

### What does the label "specific" mean at block 1?

**Reading it.** The label stands, but it should not be read as the count at that block. The strip under the figure's left panel shows why. There the covariance direction holds 0.78 of the variance, and only 0.07 of that lies between the count's classes, against 0.04 for random directions. So at block 1 it is the dominant direction of the residual and not a count direction, in keeping with the ridge's reading of 0.38 against a floor of 0.36. The two directions together hold 0.9995 of the variance there, so the residual is close to two-dimensional. Its filed meaning at this block is size or something else, and this run does not say which.

### Is block 6 a null for the covariance direction?

**Reading it.** No. It is the only block where the two directions are close in size, at 0.26 against 0.30, and the direction with no covariance turns out to carry the count anyway. Its covariance with the scalar count is zero on the fit rows, yet 0.99 of its variance lies between the count's classes, so its class means must follow a curve. This is the limit Amendment 1 noted when it said erasing a one-hot count can take up to six directions. LOW loses 3.27 log-odds to it, more than the 1.88 lost to the covariance direction, and its accuracy falls to 0.97.

P and N lose about the same, 0.208 and 0.202 of their gain. N stays over P under the covariance direction only, and under this direction the rule reads "neither". Subtracting LOW's drop supplies about half of P's share: LOW's log-odds of P's answer rise by 2.73 under the removal, so P's raw share is 0.101 and N's is 0.158. The raw share has no band, and the filed rule reads the scored share. None of this can change Amendment 1's branch, "cannot serve", which stands.

### Why does N lose anything under the covariance direction?

**Reading it.** This stays unexplained between two readings, and E70 does not separate them. At blocks 3 to 5, N loses 0.16, 0.34 and 0.49 of its gain under the covariance direction. Either N reuses the count of a in some part, or the direction leaks into the count of b. The two counts correlate at -0.50 on the `C b 40` rows the states are read on. On LOW's `C a 40` rows, the covariance direction's class share of the count of b is about 0.26. The square of that correlation is 0.25, so that share is about what the correlation alone would give a direction tracking the count of a, though the two numbers come from different rows and the correlation was not read on the `C a 40` rows. Either reading could give the loss seen here, and telling them apart needs rows where the two counts are uncorrelated, which were not run.

### The state-by-state tables

**softmax, scored share per state at 10K**

| state | blocks 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| c1b/P_1e-4_s0/10000 | 0.047 | 0.102 | 0.191 | 0.502 | 0.697 | 0.535 |
| c1b/P_1e-4_s1/10000 | 0.044 | 0.097 | 0.192 | 0.495 | 0.686 | 0.539 |
| c1b/P_1e-4_s2/10000 | 0.05 | 0.099 | 0.186 | 0.495 | 0.686 | 0.536 |
| c1b/P_1e-4_s3/10000 | 0.047 | 0.1 | 0.201 | 0.503 | 0.698 | 0.54 |
| c1b/P_1e-4_s4/10000 | 0.056 | 0.108 | 0.192 | 0.454 | 0.67 | 0.536 |
| c1b/P_1e-5_s0/10000 | 0.074 | 0.149 | 0.285 | 0.62 | 0.75 | 0.59 |
| c1b/P_1e-5_s1/10000 | 0.07 | 0.147 | 0.28 | 0.616 | 0.756 | 0.589 |
| c1b/P_1e-5_s2/10000 | 0.077 | 0.149 | 0.283 | 0.615 | 0.753 | 0.591 |
| c1b/P_1e-5_s3/10000 | 0.077 | 0.154 | 0.287 | 0.62 | 0.756 | 0.591 |
| c1b/P_1e-5_s4/10000 | 0.072 | 0.145 | 0.28 | 0.609 | 0.748 | 0.59 |
| a2/M/10000 | 0.013 | 0.044 | 0.128 | 0.23 | 0.404 | 0.525 |
| a2/N1/10000 | 0.027 | 0.044 | 0.086 | 0.177 | 0.286 | 0.462 |
| a2/N1s1/10000 | 0.028 | 0.045 | 0.076 | 0.159 | 0.248 | 0.469 |
| c1b/N_1e-4_s2/10000 | 0.031 | 0.041 | 0.076 | 0.158 | 0.284 | 0.456 |
| c1b/N_1e-4_s3/10000 | 0.03 | 0.04 | 0.079 | 0.169 | 0.283 | 0.461 |
| c1b/N_1e-4_s4/10000 | 0.024 | 0.04 | 0.077 | 0.159 | 0.276 | 0.469 |
| c1b/N_1e-5_s1/10000 | 0.015 | 0.043 | 0.12 | 0.223 | 0.4 | 0.524 |
| c1b/N_1e-5_s2/10000 | 0.012 | 0.041 | 0.12 | 0.227 | 0.408 | 0.526 |
| c1b/N_1e-5_s3/10000 | 0.01 | 0.041 | 0.126 | 0.219 | 0.396 | 0.523 |
| c1b/N_1e-5_s4/10000 | 0.013 | 0.044 | 0.13 | 0.228 | 0.406 | 0.527 |

**ridge, scored share per state at 10K**

| state | blocks 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| c1b/P_1e-4_s0/10000 | -0.006 | 0.003 | 0.004 | 0.003 | 0.005 | -0.002 |
| c1b/P_1e-4_s1/10000 | -0.006 | 0.004 | 0.005 | 0.004 | 0.007 | -0.003 |
| c1b/P_1e-4_s2/10000 | -0.006 | 0.004 | 0.005 | 0.004 | 0.005 | -0.002 |
| c1b/P_1e-4_s3/10000 | -0.006 | 0.004 | 0.004 | 0.004 | 0.005 | -0.002 |
| c1b/P_1e-4_s4/10000 | -0.006 | 0.005 | 0.005 | 0.004 | 0.005 | -0.003 |
| c1b/P_1e-5_s0/10000 | -0.007 | 0.004 | 0.005 | 0.001 | -0 | -0.004 |
| c1b/P_1e-5_s1/10000 | -0.007 | 0.004 | 0.005 | 0.001 | -0 | -0.004 |
| c1b/P_1e-5_s2/10000 | -0.008 | 0.004 | 0.005 | 0.002 | 0 | -0.003 |
| c1b/P_1e-5_s3/10000 | -0.008 | 0.003 | 0.005 | 0.002 | 0.001 | -0.003 |
| c1b/P_1e-5_s4/10000 | -0.007 | 0.004 | 0.005 | 0.001 | -0 | -0.003 |
| a2/M/10000 | 0.003 | 0.001 | -0 | 0.001 | -0.004 | -0.001 |
| a2/N1/10000 | -0.001 | -0 | 0.001 | 0.001 | 0 | -0 |
| a2/N1s1/10000 | 0 | 0.002 | 0.001 | 0.001 | -0 | -0 |
| c1b/N_1e-4_s2/10000 | -0 | 0.002 | 0.001 | 0.002 | 0.001 | -0.001 |
| c1b/N_1e-4_s3/10000 | -0 | 0.001 | 0 | 0.002 | 0.001 | -0 |
| c1b/N_1e-4_s4/10000 | 0 | 0.002 | 0 | -0 | -0.001 | -0 |
| c1b/N_1e-5_s1/10000 | 0.004 | 0.001 | -0 | 0.001 | -0.005 | -0.001 |
| c1b/N_1e-5_s2/10000 | 0.003 | 0.002 | -0 | 0.001 | -0.004 | -0.001 |
| c1b/N_1e-5_s3/10000 | 0.003 | 0.002 | -0 | 0.002 | -0.003 | -0.001 |
| c1b/N_1e-5_s4/10000 | 0.003 | 0.002 | -0.001 | 0.001 | -0.005 | -0.001 |

## What we learned

The definition's possession and function clauses behave as they should on a pair with known labels. Its causal clause, read as removing one direction, has a filed answer that stands: one direction cannot serve as ground truth here, because of the reversal at block 6.

At blocks 3 to 5 there is now a control under the gap. A direction that lies between the count's classes costs the reusing task more than the new task and costs the pretrained model heavily, while the largest direction that does not lie between them costs nothing. That supports the count's direction being causal on the pretrained model, and more used by the reusing task than the new task, at those blocks. It holds at this unmatched size, on one pair and one direction, and it is not the size-matched test that was asked for, which cannot exist with one direction at blocks 1 to 5.

It does not hold at every block. At block 1 the covariance direction is not a count direction, and at block 6 both directions lie between the classes and the reusing task is not over the new task under either. The first reading's puzzle is settled in one respect: removing the regression direction leaves the count readable, which is why it could do nothing.

The six-direction result stays provisional, since its variance was not read. So for `GOALS.md`'s question of new against existing, this pair gives a lead at three blocks and does not yet give a ground truth for scoring cheap methods on Qwen.

## How compelling is the evidence

The apparatus checks are strong for what they cover: 20 states, ten per side, five seeds and two learning rates, with the random drops reproducing the first reading's exactly. The causal reading at blocks 3 to 5 is moderate at best, for the first three reasons below.

- **The control is not matched on size.** Random directions hold about 0.004 to 0.007 of the variance, the no-covariance direction 0.09 to 0.13 at blocks 3 to 5, and the covariance direction 0.53 to 0.56 there. A smaller direction costing less is also what size alone could give. Scaling in proportion predicts more loss than was read, but a threshold effect of removing half the variance is not excluded. Block 6, the one near match on size, carries the count's classes, so it is no null.
- **One control direction.** The largest direction was chosen by size and may hold anything else the residual holds. Its zero loss at blocks 3 to 5 rules out that one direction, on LOW's variance read on `C a 40` rows and not on the states' own residuals. Its variance on the `C b 40` rows the states are read on was not read.
- **N's loss is unexplained.** The gap between the tasks sits on an N that loses up to 0.49 of its gain, with the two counts correlated at -0.50 on the rows read. This run cannot say whether that is reuse or leak.
- **Zero covariance covers the scalar count only.** LOW's head is a softmax over count tokens, and erasing a one-hot count can take up to six directions. So after this removal a linear reader of the scalar count fails, while a reader of each count class need not. LOW's accuracy at block 6 stayed 0.9995 with the scalar reading erased, and no later block remains there to read the text again. So "the refit reads the floor" does not mean the count is gone, and a task's loss here is the loss of this direction, not of the count as such. No reader of the count classes was fit after removal.
- **Blind spots kept from the first reading.** The removal misses a count held non-linearly and anything later blocks re-read from the text. It also cannot tell a direction used by both counts from a mechanism N shares.
- **Log-odds are not answers.** These states are very sure of their answers, with a mean probability above 0.999 on every 10K state, so a large log-odds drop can leave the answer in place. Under the six softmax directions the share of the gain in nats, recorded but not scored, reads 0.16 and 0.31 for P at blocks 4 and 5 against 0.002 and 0.013 for N, and about 0.001 on both at block 6. The median row on P at block 5 loses 0.01 to 0.02 nats at rate 1e-4 and 0.33 to 0.49 at rate 1e-5. Accuracy after removal was recorded only in the amendment, where answers do change at blocks 3 to 5 and do not at block 6.
- **Check.** No independent check was run on any stage, and none is claimed, so the reading above is provisional. This is one retrain of LOW, with one probe seed, and the refit reads `C a 40` rows only.

## Surprises

**A filed reversal decides the amendment.** At block 6, task N loses more of its gain than task P, 0.174 against 0.070, by more than the band. No filed number predicted N above P there. The filed lean, that the rule would hold at block 4 or 5 and not at block 6, is met to the letter, and the reversal goes beyond it. We read no cause for it, and the branch stands.

**The covariance direction does much more damage than filed.** This run measured no cause for any of the misses below.

- LOW's drop reads 1.9 to 12, against a filed 0.3 to 5 at some of blocks 3 to 6. It is above the band at all six blocks and above 5 at blocks 3 to 5.
- LOW's accuracy falls to 0.573, 0.371 and 0.3605 at blocks 3 to 5, against a filed at or above about 0.7.
- P's share is over the filed ceiling of the six-direction means at blocks 2 to 5: 0.160, 0.427, 0.779 and 0.884 against 0.125, 0.238, 0.553 and 0.720. So one direction cost P more than six did at those blocks. Block 6 reads 0.070 against 0.564.
- N's share is over its filed values at blocks 4 and 5: 0.335 and 0.493 against about 0.20 and 0.34. Block 6 reads 0.174, within its filed 0.49.
- The share in nats is over its filed values at blocks 4 and 5 on both tasks: P 0.650 and 0.833 against 0.155 and 0.308, N 0.021 and 0.071 against 0.002 and 0.013.
- Task accuracy falls below the filed 0.9 on the rate 1e-4 states. P reads 0.78 to 0.88 at block 3, about 0.50 at block 4 and 0.31 to 0.47 at block 5, and N reads 0.82 to 0.93 at block 5. The rate 1e-5 states, filed as possibly under 0.9 on P at blocks 4 and 5, also fall under it on P at blocks 2 and 3 and on N at block 5.
- The variance along the regression direction reads 0.016 at block 4, against a filed under 1/192, about 0.005.

**What the amendment read as filed.** The refits, the cosine, the random controls and P's and N's order at blocks 4 and 5 all read as filed, as the scorecard shows.

**The second amendment: the direction with no covariance carries the count at block 6.** We filed no lean for its class share. It reads 0.99 at block 6 against under 0.02 at blocks 1 to 5, and LOW loses more to it there than to the covariance direction. The filing named a curved component as possible, and it appeared only at the last block, which takes away the only near match on size. This run measured no cause for that.

**The second amendment's other misses, each narrow.**

- P's share under the no-covariance direction was filed to be above the band at most blocks, and it is above only at blocks 2 and 6.
- The variance held was leaned at 0.1 to 0.3, and it reads 0.096 and 0.086 at blocks 4 and 5.
- LOW's accuracy under it was filed above the covariance direction's where it holds less variance, and at block 6 it reads 0.9715 against 0.9995.
- The refit at block 2 was filed near 0.9975, and it reads 0.9165.
- The random draws' class share was filed between the two directions, and at block 6 it reads 0.78, under both.

**In the first reading: a filed reversal, and a filed miss, on the headline.** We predicted the one-direction rule would hold, and instead it does not separate, with N above P at block 1 by half a percent. The size is tiny and the removal has no effect on LOW, so we read no mechanism into it.

**In the first reading: the direction that reads the count does nothing when removed.** A regression direction reads the count at accuracy 1 from block 3 and supports a perfect lookup for P, yet taking it away leaves LOW's log-odds inside the random band. The amendment answers this: the count stays readable without that direction.

**In the first reading: N loses much more than predicted under the six directions.** N reads 0.34 at block 5, against a predicted value near 0.06.

Narrow: the softmax probe read up to 0.06 above its earlier run at blocks 2 and 3, and neither probe read consistently lower on `C b 40` rows at early blocks, as we had predicted.

## What next

Nothing is proposed. E70 has used its two amendments and ends as it reads: a third would be a new plan, and none is proposed. What stays open is listed under "Not run", and the first item there says which result would bear on N's loss.

## Scorecard

Verdicts are the filed rules' own. A probe's "no teeth" means its drop on LOW did not exceed its random band.

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| R1a | LOW retrains to its old count | within 0.01 of 0.99975 | 0.99975 | holds |
| R1b | a removal costs LOW, or stop | drop above band at some block, some probe | ridge at none, softmax at all six | no stop; ridge no teeth |
| R2 ridge, headline | P loses more than N | smallest P minus largest N above band, at a block from 1 to 5 | gaps within 0.012, none above band | does not separate; lean missed |
| R2 reversal | N above P | smallest N minus largest P above band | block 1, 0.0050 against 0.0042 | reversal, filed, half a percent |
| R2 softmax, beside it | P loses more than N | as above | gaps 0.052, 0.056, 0.224, 0.262 at blocks 2-5, bands 0.041, 0.041, 0.045, 0.063 | separates |
| Lean, P | high late, over 0.5 at block 6 | mean, 10K | softmax 0.56; ridge -0.003 | met on softmax, not on ridge |
| Lean, N | near the band at blocks 1-5 | mean, 10K | softmax 0.10, 0.20, 0.34 at blocks 3-5, bands 0.041, 0.045, 0.063 | missed on softmax |
| Lean, LOW | above band at least at block 6 | drop against band | softmax 6.38 against 0.36; ridge -0.02 against 0.08 | met on softmax, not on ridge |
| Lookup | P near probe, N near floor | 10K | P 1.00 at block 6; N 0.387, floor 0.357 | as expected |
| Task accuracy | close to the earlier stage | logged | 10K all 1.0, within 0.0005; 1K within 0.015 | as expected |

**Amendment 1.** Filed rules in `results/predictions_e70_a1.json`.

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| R0 | same rows, file and code as the first reading | ridge within one row, random drops within 1% of the band | ridge and floor equal, largest random difference 0 | holds |
| R1 | LOW loses log-odds | drop above band, one-sided | all six blocks | teeth; see the size caveat |
| R3 | the removal does what it claims | refit under the midpoint of floor and before, blocks 2-6 | 0.359 to 0.3705 | holds |
| R2 | P loses more than N | smallest P minus largest N above band | blocks 1, 3, 4, 5 hold; block 2 neither; block 6 reversal | as read |
| Tree | first match wins | a reversal at any read block | reversal at block 6 | **cannot serve** |
| Lean, LOW drop | about 0.3 to 5, some of blocks 3-6 | | 1.88 to 11.96, all six above band | missed |
| Lean, LOW accuracy | at or above about 0.7 | | 0.3605 at lowest | missed at blocks 3-5 |
| Lean, refit after covariance direction | 0.33 to 0.40 | | 0.3575 to 0.3705 | as expected |
| Lean, refit after regression direction | above 0.9 at blocks 3-6, near 0.9975 at block 2 | | 1, and 0.993 | as expected |
| Lean, variance | covariance above regression and above 1/192 at blocks 2-6; regression under 1/192 at blocks 2-4 | | first as filed; regression 0.016 at block 4 | missed at block 4 only |
| Lean, cosine | size below 0.5 at blocks 2-6 | | at most 0.32 | as expected |
| Lean, P | above band, below six-direction means at most blocks, highest at blocks 4-5 | | above the means at blocks 2-5, below at block 6; highest at 4-5 | ceiling missed |
| Lean, N | under P at blocks 4-5, above band at 4-6, up to about 0.20, 0.34, 0.49 | | under P; above band; 0.335, 0.493, 0.174 | over at blocks 4-5 |
| Lean, R2 | holds at 4 or 5, not at 6 | | holds at 4 and 5; reversal at 6 | met to the letter; reversal not predicted |
| Lean, nats | at most about P 0.155, 0.308 and N 0.002, 0.013 at blocks 4, 5 | | P 0.650, 0.833; N 0.021, 0.071 | missed |
| Lean, state accuracy | at or above 0.9 on 1e-4 states; 1e-5 P may fall at blocks 4-5 | | 1e-4 P 0.31 to 0.47 at block 5 | missed |

**Amendment 2.** Filed rules in `results/predictions_e70_a2.json`.

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| R0 | same rows, file and code as the first reading | ridge within one row, random drops within 1% of the band; Amendment 1's rule held at 1, 3, 4, 5 | ridge and floor equal, largest random difference 0 | holds |
| R4 | the direction has no covariance with the count on the fit rows | at most 1e-3 of the covariance direction's at every block | at most 1.1e-8 | holds |
| R3' | the count stays linearly readable after removal | refit above the midpoint of floor and before, blocks 2-6 | 0.9165, 0.9975, 1, 1, 0.9995 | holds |
| Teeth | LOW's drop against the band, printed, stops nothing | one-sided | above at blocks 1, 2, 6; not at 3, 4, 5 | as read |
| R2 | P loses more than N under this direction | smallest P minus largest N above band | neither at all six blocks | does not separate |
| Label, block 1 | specific | R2 not holding; P under the covariance direction over P here by more than band | 0.133 against -0.0005, band 0.0042 | specific; not read as the count |
| Label, block 3 | specific | as above | 0.427 against 0.0009, band 0.0125 | specific |
| Label, block 4 | specific | as above | 0.779 against -0.0051, band 0.0191 | specific |
| Label, block 5 | specific | as above | 0.884 against -0.0010, band 0.0114 | specific |
| Tree | first match wins | a specific block and no size block | specific at 1, 3, 4, 5 | **count-specific at this size** |
| Block 6, apart | does N stay over P? | printed, unscored | neither; N minus P -0.006, against 0.104 under the covariance direction | unable to change "cannot serve" |
| Lean, branch | count-specific at this size, or mixed | | count-specific at this size | as leaned |
| Lean, variance | above 1/192, at most what the covariance direction leaves, 0.1 to 0.3 at blocks 1-5 | | 0.224, 0.265, 0.134, 0.096, 0.086, and 0.264 at block 6 | bounds as filed; lean narrowly missed at 4, 5 |
| Lean, LOW drop | above band at some blocks, under the covariance direction's at 1-5 | | above at 1, 2, 6; under at 1-5 | as expected |
| Lean, LOW accuracy | at or above about 0.7; above the covariance direction's where it holds less variance | | at least 0.9715; below at block 6 | first as expected; second missed at block 6 |
| Lean, covariance | fit rows about 1e-7, held rows 0.1 or less, cosine under 1e-5 | | at most 1.1e-8, 0.016, 4.2e-9 | as expected |
| Control, class share | covariance direction above 0.5 at 3-5, low at 1; random between the two | | 0.93, 0.98, 0.98; 0.07; random between at 1-5, under both at 6 | as expected but for random at 6 |
| Lean, refit | at or above 0.9 at 3-6, near 0.9975 at 2, 0.33 to 0.40 at 1 | | 0.9975, 1, 1, 0.9995; 0.9165; 0.3815 | narrowly missed at 2 |
| Control, covariance direction again | LOW's drop equal to Amendment 1's | printed, no rule | equal at all six blocks | as expected |
| Control, random | Amendment 1's bands and variance shares | | equal | as expected |
| Lean, P | below Amendment 1's at 1-5, above band at most blocks | | below at 1-5; above band at 2 and 6 only | second missed |
| Lean, N, block 6, class share of the direction | none filed | | N 0.0046, 0.0094, 0.0009, -0.0001, 0.0048, 0.2020; class share 0.99 at block 6 | no lean |

## Provenance

Plan `reports/plans/2026-11-03_plan_e70-existing-new-counting.md`. Rules `experiments/pcfg_anchors/results/predictions_e70.json`. Code `experiments/pcfg_anchors/e70.py`, `src/commute/removal.py`. Results `experiments/pcfg_anchors/results/e70.json`, commit a6103fc. Figures `experiments/pcfg_anchors/figures/e70_{low,ridge,softmax}.png` from `plot_e70.py`. Report `reports/analyses/2026-11-03_analysis_e70-existing-new-counting.md`. MATS L40, `debug`, jobs 50203, 50242, 50298, 50311, 50361, 50363, 11:17 to 13:17 on 2026-11-03, $0.

Amendment 1: plan section "Amendment 1" in the same plan. Rules `experiments/pcfg_anchors/results/predictions_e70_a1.json`. Code `experiments/pcfg_anchors/e70_a1.py`, `src/commute/removal.py`, `cluster/e70_a1.sbatch`. Results `experiments/pcfg_anchors/results/e70_a1.json`, commit afc1191. Log `cluster/logs/commute-e70-a1-50499.out`. Figures `experiments/pcfg_anchors/figures/e70_a1_{low,tasks}.png` from `plot_e70_a1.py`. MATS L40, `debug`, job 50499, exit 0, 15:41 to 15:56 on 2026-11-03, peak host memory 1.4 GB, $0.

Amendment 2: plan section "Amendment 2" in the same plan, ask 17 taken by default. Rules `experiments/pcfg_anchors/results/predictions_e70_a2.json`. Code `experiments/pcfg_anchors/e70_a2.py`, `src/commute/removal.py`, `cluster/e70_a2.sbatch`, commit c594bc9. Results `experiments/pcfg_anchors/results/e70_a2.json`, commit 71d3975. Log `cluster/logs/commute-e70-a2-50643.out`, with a copy at `experiments/pcfg_anchors/results/large/e70/e70_a2.log`. Figures `experiments/pcfg_anchors/figures/e70_a2_{low,tasks}.png` from `plot_e70_a2.py`. MATS L40, `debug`, job 50643, exit 0, 18:21 to 18:36 on 2026-11-03, peak host memory 1.4 GB, $0.

## Not run

- **Any removal on rows where the two counts are uncorrelated, so the -0.50 correlation is untested.** A fall in N there would support the leak reading of its loss, and no fall would rule it out. This would need a new plan, and none is proposed.
- A size-matched control: no single direction with no covariance with the count can match at blocks 1 to 5, and at block 6 the near match carries the count's classes.
- A removal of higher rank, with no covariance, holding the covariance direction's variance, which would bear on a threshold effect of size.
- Several large random directions at right angles to the covariance direction, and a direction with no class share at all with the count.
- The variance share on the states' own residuals; every variance share here is LOW's on `C a 40` rows.
- The correlation of the two counts on `C a 40` rows, where the class share of the count of b is read.
- The variance held by the six softmax directions, which were never read against their band for size.
- LEACE's oblique map, and an erasure of the one-hot count of rank up to 6.
- A reader of the count classes after removal, and a refit on `C b 40` rows.
- Either amendment on the 1K states.
- Raw-share bands: random draws were scored only as the subtracted share.
- Pruning and reverse fine-tuning on these retrained states; the earlier stage's readings stand and are quoted, not rerun.
- A second LOW or a second probe seed; any retraining in either amendment.
- VPD and Qwen.
- Carrying the lookup to count values it never saw: no held-out rule is filed.
- An independent check.

## Departures from the literature

- **Possession.** Jain et al. 2024 threshold a softmax probe for possession. We report by block with no threshold, by the user's instruction, against "filed before any run" in `decisions.md`, and use a ridge regression as the headline, because a removal needs a direction.
- **Causal test.** Jain et al. use pruning and reverse fine-tuning. We remove a probe's direction or subspace at `=`, as in `decisions.md`'s mechanistic row, because this pair's labels are known by construction.
- **Shuffled-label control.** Hewitt and Liang 2019 use random labels per input type. We permute labels within each batch, since these inputs carry no information about them.
- **Choice of direction.** Haufe et al. 2014, cited from memory and not re-read, take a linear model's activation pattern as the covariance of the input with the model's output, against its filter, the weights. We use the covariance with the label, the count. The two agree up to scale where the ridge reads the count, blocks 2 to 6, and differ at block 1. We do this because the removal is defined on the count, not on a probe.
- **Erasure.** Belrose et al. 2023, LEACE, cited from memory and not re-read, erase a one-hot concept with an oblique map that subtracts along the covariance and reads with the whitened filter. We remove the covariance direction by an orthogonal removal, on a scalar count, at `=` only, with later blocks free to re-read the text. We do this to keep rank 1 and the first reading's removal and rules. It zeroes the same cross-covariance for the scalar only, and the oblique map is not run. The second amendment removes its direction the same way, since only the direction may differ.
- **A large null direction.** Random-direction nulls for removal, as in Elazar et al. 2021 and Belrose et al. 2023, both cited from memory and not re-read, control for the removal's rank and not for the variance it takes. The second amendment uses the top principal component after the covariance direction as a null for variance taken, with zero scalar covariance by construction and no match on size. We do this because the covariance direction holds up to 170 times a random direction's variance, and no exact match at right angles to it exists at blocks 1 to 5.
