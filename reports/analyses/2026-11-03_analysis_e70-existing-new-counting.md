# Does removing the old count hurt the reusing task more than the new one?

E70, benchmark line, 2026-11-03. Plan `reports/plans/2026-11-03_plan_e70-existing-new-counting.md`. Rules `experiments/pcfg_anchors/results/predictions_e70.json`. Results `results/e70.json`, commit a6103fc. Notebook `notebooks/E70_existing_new_counting.ipynb`.

We trained probes for the count of a on the pretrained counting model, then removed what they read from two fine-tuned tasks: one built to reuse that count, one built to need a new count.
Removing the one direction the task named did nothing, even on the pretrained model, while removing six directions hurt the pretrained model and cost the reusing task more at four of six blocks.
So the one-direction causal test of new against strengthened capabilities stays unanswered, and the wider removal gives only a provisional sign that the test can work, by degree.

## The answer

Only with a six-direction removal, and that is not the test we set out to run. The one direction the regression probe finds reads the count perfectly, yet removing it costs the pretrained model nothing and costs both fine-tuned tasks under one percent of their gain. So the filed headline, no separation, says nothing about the definition. Removing the softmax probe's six directions does hurt the pretrained model, and costs the reusing task more at blocks 2 to 5, though the new task loses far more than predicted.

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

## What we expected, and why

We expected the removal to work on LOW, at least at the last block, since a readout that reads the count should be using what the model uses.

We expected the rule to hold. P's answer is a function of the old count, so removing the count should cost P a lot: over half its gain at block 6. N's answer does not depend on the count of a, so N should sit near the band at blocks 1 to 5, and might rise at block 6, where every answer shares the output head.

We expected the function test to split the pair cleanly. A table from LOW's reading to the state's answer should score near the probe's accuracy on P and near its floor on N.

The plan named one blind spot in advance: the count of a and the count of b are correlated on these rows, so a direction shared between them would cost N as well.

## What we observed

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

So one direction that reads the count perfectly can be taken away with no effect on the model that holds the count. Why is unexplained and open. Whatever follows on the ridge probe has no working control under it.

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

The definition's possession and function clauses behave as they should on a pair with known labels. Its causal clause, read as the removal of one regression direction, gave no reading, because that removal has no effect even on the model that holds the count. A wider removal of six softmax directions has an effect on LOW and costs P more than N at blocks 2 to 5, with N also losing a lot. So for `GOALS.md`'s question of new against existing, the one-direction test on known labels remains unanswered. The six-direction result is a provisional sign that a causal removal can carry the distinction, by degree, not yet a ground truth for scoring cheap methods on Qwen.

## How compelling is the evidence

The controls are strong for what they cover: 20 states, ten per side, five seeds and two learning rates, with bands from the same job.

- **Blind spots.** The removal misses a count held non-linearly and anything later blocks re-read from the text. It also cannot tell a direction used by both counts from a mechanism N shares.
- **Log-odds are not answers.** These states are very sure of their answers, with a mean probability above 0.999 on every 10K state, so a large log-odds drop can leave the answer in place. The share of the gain in nats, recorded but not scored, reads 0.16 and 0.31 for P at blocks 4 and 5 against 0.002 and 0.013 for N. It reads about 0.001 on both at block 6. The median row loses only 0.2 nats on P at block 5. No accuracy after removal was recorded. This reading is provisional and has no filed rule.
- **Check.** No independent check was run, and none is claimed. This is one retrain of LOW, with one probe seed.

## Surprises

**A filed reversal, and a filed miss, on the headline.** We predicted the one-direction rule would hold, and instead it does not separate, with N above P at block 1 by half a percent. The size is tiny and the removal has no effect on LOW, so we read no mechanism into it.

**The direction that reads the count does nothing when removed.** A regression direction reads the count at accuracy 1 from block 3 and supports a perfect lookup for P, yet taking it away leaves LOW's log-odds inside the random band. We cannot explain this.

**N loses much more than predicted under the six directions.** N reads 0.34 at block 5, against a predicted value near 0.06.

Narrow: the softmax probe read up to 0.06 above its earlier run at blocks 2 and 3, and neither probe read consistently lower on `C b 40` rows at early blocks, as we had predicted.

## What next

- **Remove a direction chosen by where the count actually varies in the residual,** the covariance of the residual with the count, not the regression weights, on LOW first against its random band, about 15 minutes at $0. If LOW loses log-odds, read P and N: separation would give the one-direction reading the task named, and no separation would mean one direction cannot serve as ground truth here. If LOW loses nothing again, the one-direction test is closed and the six-direction result is the only reading left.
- **Test N's rise.** Repeat the six-direction removal on rows where the two counts are uncorrelated. A fall in N would support the shared-direction reading, and no fall would rule it out.
- **Record accuracy after removal,** so a share of log-odds can be read as answers lost or kept.
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

## Provenance

Plan `reports/plans/2026-11-03_plan_e70-existing-new-counting.md`. Rules `experiments/pcfg_anchors/results/predictions_e70.json`. Code `experiments/pcfg_anchors/e70.py`, `src/commute/removal.py`. Results `experiments/pcfg_anchors/results/e70.json`, commit a6103fc. Figures `experiments/pcfg_anchors/figures/e70_{low,ridge,softmax}.png` from `plot_e70.py`. Report `reports/analyses/2026-11-03_analysis_e70-existing-new-counting.md`. MATS L40, `debug`, jobs 50203, 50242, 50298, 50311, 50361, 50363, 11:17 to 13:17 on 2026-11-03, $0.

## Not run

- A one-direction removal chosen by covariance, or any direction other than the ridge weights.
- The six-direction removal on rows where the two counts are uncorrelated, so the -0.50 correlation is untested.
- Accuracy after removal.
- Raw-share bands: random draws were scored only as the subtracted share.
- Pruning and reverse fine-tuning on these retrained states; the earlier stage's readings stand and are quoted, not rerun.
- A second LOW or a second probe seed.
- VPD and Qwen.
- Carrying the lookup to count values it never saw: no held-out rule is filed.
- An independent check.

## Departures from the literature

- **Possession.** Jain et al. 2024 threshold a softmax probe for possession. We report by block with no threshold, by the user's instruction, against "filed before any run" in `decisions.md`, and use a ridge regression as the headline, because a removal needs a direction.
- **Causal test.** Jain et al. use pruning and reverse fine-tuning. We remove a probe's direction or subspace at `=`, as in `decisions.md`'s mechanistic row, because this pair's labels are known by construction.
- **Shuffled-label control.** Hewitt and Liang 2019 use random labels per input type. We permute labels within each batch, since these inputs carry no information about them.

