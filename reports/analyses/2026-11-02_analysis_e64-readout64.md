# Does a block before the last hold the half-known values, and does the fine-tune's gain run through them?

We trained a linear readout on Pile text at each block of the unchanged model, asked which keys it holds, and removed that direction from twelve fine-tuned models. By the filed rule the run is unreadable: the readout missed its fit bar and the removal did not separate the two kinds. For the goal of telling strengthened capabilities from new ones, the pair's label gains no causal footing here, and the next step is the user's.

**The filed verdict.** The branch reads, as filed: "readout fit missed and P1 elicit clause or P2 missed (P1 claim; P2 fail): unreadable; one training run was the cap; tell the user". Everything under "Provisional readings" is separate from it and changes nothing in it. Two of those readings matter most. Block 0's holding tracks how often the training sample pairs each key with its value. And this readout's row has not been shown able to read a large share at any block, so a low share means little.

**"Elicit" in this report** means built to elicit: how the table was made, from W0's own guesses. It is not the causal label of `decisions.md` row 112, and under the row "Held-out rule" the causal check is what would give that label. This run's causal check came back unreadable, so the pair carries no causal label from it.

## What we were trying to test

E54's pair is two lookup tables over the same 32 keys, each asked after a marker token. The table built to elicit maps each key to a value W0 already ranks highly after that marker and key. The cipher maps the same keys to values W0 gives almost no chance. Twelve fine-tuned weight changes were already saved: five built to elicit, five ciphers at the matched step, and two ciphers at their own step.

**The readout.** One linear map per block takes the residual at one position and predicts the next token among the pair's 64 values. It trains on Pile windows whose next token is one of the 64, at most 2,000 per value, and never on the tables. The position is index 20 of a 41-token window, which on a task window is the key token itself, one before the answer. A second readout, capped at 241 windows per value, is there to test whether row quality drives the removal.

**Holding a key, the plan's "possessing".** A block holds a key if the key's value has median rank 5 or better among the 64 over the key's windows, and if its probability at those windows is at least 3 times its median at other keys' windows. W0's own output goes through the same rule first, and gives the reference set of 20 elicit keys and no cipher keys. A block holds the elicit map if it holds 3/4 of those 20. It holds the cipher if it holds at least 6 cipher keys.

**The removal share.** At the earliest block before the last that holds the elicit map on the fit half, the readout row of the window's target value is removed from the residual at that position: $h \leftarrow h - (h \cdot u)u$, with $u$ the row made unit length. The share is $A = (\text{state's log-odds drop} - \text{W0's drop}) / \text{state's log-odds gain}$. Zero means the removal costs the state no more than it costs W0, and 1 means it takes away the whole gain.

**The controls.** W0 must lose at least 0.3 nats under the target row, and more than under a wrong value's row, on the windows of its 20 keys. A random unit direction must give a share within 0.05 on every state. The last block's readout must reproduce W0's held set. The last-block removal was reported and expected near 1.

## What we expected, and why

We expected the elicit map to be held before the output and the cipher not, because W0 already knows these values in part. We expected the removal to cost elicit fine-tunes a larger share of their gain than ciphers, by at least 0.2, because a fine-tune built to elicit should use what W0 already has. We expected the last block to reproduce W0's held set almost by construction. The plan's null was that the values are held only at the last block, or that the removal costs both kinds alike.

## What we observed

### Did the readout fit? No, by the bar as run

![fit](../experiments/readout64/figures/fit_loss.png)

**Reading it.** The fit missed. The last block's held-out loss is 1.44 nats, against 1.27 for W0 moved to the sample's prior, so it sits 0.18 above W0 where the bar allows 0.10. Held-in loss is 1.10, so the readout over-fits somewhat.

**Under the first-filed bar this fit would have passed, and the branch would have read differently.** That bar compared with W0 at its natural prior: 1.83 plus 0.10, so 1.93, and 1.44 clears it. The branch code with a passing fit returns "P2 fails: r's reading of this pair has no causal footing; the line pauses on the 67M and E55 C2 decides r's standing". The bar moved on review finding F3, filed in commit `fcb049f` at 13:22:43 and before the job started at 13:23:13, because W0's natural loss on a capped sample is inflated by the values over the cap and the readout gets that margin free. The amendment preceded every reading: job 48611 is the only E64 job in `cluster/logs/jobs.log`, and nothing had run on the model or a saved state before it. The verdict is the one the bar as run gives. That other branch would have over-read this run, because the share has not been shown able to read high, as the last-block section explains. The fit bar is not the only thing between this run and that conclusion.

| held-out loss, nats | block 0 | block 1 | block 2 | block 3 |
|---|---|---|---|---|
| scored readout | 2.352 | 2.009 | 1.640 | 1.441 |
| capped readout, reported | 2.604 | 2.277 | 1.930 | 1.815 |
| scored readout, held-in | 2.132 | 1.759 | 1.365 | 1.104 |
| capped readout, held-in | 1.946 | 1.458 | 0.973 | 0.638 |

W0 on the same held-out windows reads 1.835 at its natural prior and 1.266 at the sample's prior. Bars: 1.366 as run, 1.935 first-filed. The capped readout has no bar.

### Which blocks hold the map? Every block for elicit, none for the cipher

![possession](../experiments/readout64/figures/possession.png)

**Reading it.** Every block holds 17 of W0's 20 elicit keys, above the bar of 15, and no block holds a single cipher key. The surprise is block 0: it holds more keys in all than the last block does, and its median rank is 1 where the last block's is 2. So the filed rule says the map is held at the block closest to the input. The dotted-and-dashed line is a table of next-token counts with no model, and the provisional readings take it up. The cipher clause has no power at block 0: no cipher pair occurs anywhere in the fit rows, so a readout that learns the lookup cannot hold a cipher key whatever W0 does.

| block | elicit keys held, of 32 | of W0's 20 | outside W0's 20 | median rank | median P ratio | cipher keys held | cipher median rank | cipher median P ratio |
|---|---|---|---|---|---|---|---|---|
| 0 | 24 | 17 | 7 | 1 | 51.5 | 0 | 45 | 0.99 |
| 1 | 24 | 17 | 7 | 1 | 35.0 | 0 | 43 | 0.90 |
| 2 | 23 | 17 | 6 | 2 | 14.3 | 0 | 44 | 0.75 |
| 3 | 20 | 17 | 3 | 2 | 14.6 | 0 | 44 | 0.63 |

The fit half agrees, with 15 of 17 at blocks 0 to 2 and 16 of 17 at block 3, cipher 0. So the removal block is block 0. The rule takes the earliest block, so the removal ran where holding follows the pair counts most closely.

### Does removing the row cost elicit fine-tunes more of their gain? No: elicit sits below the cipher

![removal](../experiments/readout64/figures/removal.png)

**Reading it.** The removal did not separate the kinds in the predicted direction. Elicit shares run from -0.042 to -0.027, cipher shares from 0.009 to 0.014, so the gap is -0.045 where the rule needs at least +0.2. The capped readout reads the same way, with a gap of -0.037. A cipher at its own step sits within 0.01 of its matched step. All twelve shares are small beside each state's gain.

| kind | share, scored | share, capped | do-nothing share | state's drop | W0's drop | drop ratio | gain, log-odds | log P lost, mean | log P lost, median |
|---|---|---|---|---|---|---|---|---|---|
| elicit, 5 | -0.042 to -0.027 | -0.033 to -0.026 | -0.0032 to -0.0019 | 1.03 to 1.19 | 1.49 | 0.69 to 0.80 | 10.8 to 11.2 | 0.06 to 0.15 | 0.008 to 0.029 |
| cipher, matched, 5 | 0.009 to 0.014 | 0.006 to 0.009 | -0.0008 to 0.0009 | -0.23 to -0.15 | -0.32 | 0.48 to 0.70 | 10.6 to 12.1 | -0.19 to -0.11 | -0.18 to -0.04 |
| cipher, own step, 2 | 0.014 to 0.016 | 0.007 | 0.0009 to 0.0023 | -0.04 to 0.00 | -0.32 | -0.00 to 0.12 | 20.3 to 20.7 | 0.00 | 0.00 |

Raw share, recorded and not filed, with W0's drop not subtracted: elicit 0.093 to 0.110, cipher -0.021 to -0.013, smallest gap +0.106. It is under the same 0.2, and it carries W0's own dependence on the row, which is what the filed share subtracts. By the rule's means the raw gap is +0.119 and the W0 term moves it by -0.164, which gives -0.045. Subtracting W0 is still right for the question asked, since the state must depend on the row beyond W0.

### What does the removal do? It works on W0, and the states lose less than W0 loses

![drops](../experiments/readout64/figures/drops.png)

**Reading it.** W0's control passed cleanly at block 0: its median log P of the target falls 1.82 nats under the target row, 0.00005 under a wrong value's row and 0.002 under a random direction, on 31 windows. That direction matters to W0's answer on elicit windows. The effect is specific to the row, which is tied one-to-one to the key, so value and key are not separated. A row that detects the key token would pass this control as well. The wrong row's median also sits below the random direction's, and a median near 0 can hide moves both ways. On cipher windows the same removal raises W0's answer by 0.39 nats, so the cipher arm has no positive control. Each fine-tuned state moves the way W0 moves, by less.

### Can this readout's row take the gain away at the output? No

![last block](../experiments/readout64/figures/last_block.png)

**Reading it.** The reported last-block removal missed its expected value of near 1. It reads 0.50 to 0.53 for elicit, 0.34 to 0.36 for the matched cipher and 0.31 to 0.32 at the cipher's own step. So this readout's row does not remove the whole gain at the last block. The head's own row was never tried, so this says nothing about rows in general. The gap at the last block is 0.176 by the separation rule, with a range of 0.036, not scored and under 0.2. With W0's drop not subtracted the shares are 0.76 to 0.80 and 0.17 to 0.18. The twelve states change only the 24 attention and MLP matrices, so head and final norm are W0's. Every log-odds gain therefore passes through this residual at the last block, and a share short of 1 there cannot be the gain taking another route. The plan's premise was false: the row predicts Pile's next token under a capped prior, and nothing makes it the head's row. The row's angle to the head was not measured, so the reason is open.

**What this implies at earlier blocks.** The share is the gain's share times an unknown efficacy of the row, and that efficacy may differ by kind. The two kinds already differ by 0.176 at the last block, where a route difference is impossible, so a gap of that size can come from row geometry. Dividing an earlier share by the last-block share is not valid either. So an earlier-block removal reads one way only: a high share above a wrong-row reference would be evidence of use, and a low or negative share is no evidence of absence.

## Provisional readings, separate from the filed verdict

None of these was filed, none changes a verdict, and none is a tested cause.

### 1. Is block 0's holding explained by the prior tilt?

The filed line on what the rule is blind to names a tilt: common elicit values train on up to 2,000 windows and cipher values on about 350, worth up to about 1.7 nats in rank. The saved data do not settle the tilt, and neither did the filed capped-readout check below. The tilt is visible, and it never reaches the bar here.

![tilt](../experiments/readout64/figures/tilt.png)

- **Same window count, different result.** Six elicit values have window counts inside the cipher's range of 241 to 649. Block 0 holds five of them at rank 1, where it holds none of the 32 cipher keys. The sixth, value of key 4577, reads rank 39. Six keys is few, and this split is not independent of the count table below: the five have pair counts of 155, 25, 14, 15 and 15, and the sixth has 0.
- **Training count and rank.** The readout sees capped counts, and 19 elicit values train at exactly 2,000. Block 0's rank correlates 0.10 with the capped count over all 32 elicit keys. Over the 13 values whose count varies it reads -0.29, the sign a tilt predicts. On the cipher, where no count is capped, it reads -0.53, and -0.58 at the last block, yet the best cipher rank is 10. An earlier 0.25 here used counts before the cap, which the readout never saw.
- **The two may work together.** Of the 4 fit keys held with a pair count of 0 or 1, three have values at the cap: 2515, 4367 and 4447.
- **Moving W0's prior hardly moves the cipher.** W0 at the training prior shifts common values down by up to 5.2 nats. The cipher's median rank improves by 1, at most by 7, the best cipher key reaches rank 13 and none gets to 5. This is W0's output, not the readout, so it is only an indication.
- **The specificity ratio separates the kinds too.** A prior adds about the same amount to a value's log-probability at every window, so it should largely cancel in the ratio. Block 0's median ratio is 51.5 for elicit and 0.99 for the cipher, with 29 and 6 keys over the bar of 3. A worse-fitted row could also lower the ratio, which is why the count comparison above matters more.

**A plainer account, known and filed too narrowly.** The readout reads the residual at the key token, so at block 0 it can see which token is there. We counted, on the readout's own training sample and with no model, how often each key is directly followed by each of the 64 values (`experiments/readout64/bigram_check.py`, post-hoc, `results/pile_bigrams.json`). E59's "Not run" list named this key-only table, and the filed blind line names a bigram-like early block, but only for keys outside W0's set.

![lookup](../experiments/readout64/figures/lookup.png)

- **Every cipher pair occurs zero times:** in the training sample, and in all of the fit rows before the cap. Elicit pairs occur 459 times in the sample and 1,906 times in the fit rows, and 5 of the 32 are absent from the sample.
- **Pair count nearly decides holding at block 0.** 21 of the 32 elicit pairs occur at least twice, and no cipher pair occurs at all. Block 0 holds 20 of the 20 fit keys with a pair count of at least 2, against 4 of the 10 other fit keys. The base is small: 810 of the 57,345 windows have a key token at the read position, and 459 of those are pairs.
- **The count table is a guide and not a pass.** Through the same rule it holds 15 of W0's 20 at pseudocount 1, and 17, 16 and 11 at 0.1, 0.5 and 5, one fewer each with unfit key 1495 dropped; cipher 0 at all four. Counts tie at 0 where logits never do, so only the specificity clause decides, and the pseudocount has no source. The like-for-like figure at pseudocount 1 is 14, under the bar.
- **The table and block 0 agree on which keys.** 20 of block 0's 24 keys are in the table's 21, where chance is 15.8. For W0's output it is 15 of its 20, where chance is 13.1. The rank correlation with the table's rank is 0.86 at block 0, 0.77 at block 1, 0.46 at block 2 and 0.52 at block 3. W0's output reads only 0.14.
- **Two keys go beyond the pairs, and one may be the prior.** Of the 5 elicit pairs absent from the sample, block 0 ranks 2 at 5 or better. Key 4367's pair occurs 20 times in the fit rows, dropped by the cap, and its value is at the cap. Only key 3541's pair is absent from all fit rows. Nearby tokens were not checked.

**Provisional reading.** The filed clause passes as filed, but block 0's pass is not evidence that W0 computes the map there. It fits the residual carrying the key's identity and the sample pairing the two. Blocks 1 and 2 are not clear of this either, with overlaps of 19 and 18. A correlation cannot show the readout used those windows, and the elicit values were chosen from W0's guesses, so Pile pairs and W0's guesses share a cause. These numbers cannot separate "block 0 holds the map" from "the key's token identity plus Pile co-occurrence gives it". The specificity guard stops frequency, and it does not stop a one-token co-occurrence.

**The capped-readout check: no_power, as filed.** Job 48707 put the saved capped readout through the holding rule at every block, $0, 22 s, exploratory and reported only (`experiments/readout64/capped.py`, filing `results/predictions_capped.json`, result `results/capped.json`). The filed reading is `no_power`, so it settles neither the tilt nor the elicit case. The control needed the last block at 15 of W0's 20 or more, and it read 13. Under that case the filing does not read block 0's elicit count at all.

![capped](../experiments/readout64/figures/capped.png)

**Beside it, unfiled.** At equal position counts the cipher stays at 1 of 32 at every block, the same key each time, 1246. Elicit reads 17, 17, 16 and 13 of W0's 20 at blocks 0 to 3. Elicit's 17 at block 0 is above every like-for-like table on the capped sample, which runs from 4 to 14 across the pseudocount, and the filing does not read it. W0 at the same flat prior holds 20 elicit keys and no cipher key, reported as the reference. Equal counts are 62 of the 64 values, since two elicit values have 147 and 26 windows.

| capped readout, all windows | block 0 | block 1 | block 2 | block 3 |
|---|---|---|---|---|
| elicit keys of W0's 20 | 17 | 17 | 16 | 13 |
| all elicit keys held, of 32 | 23 | 22 | 22 | 14 |
| cipher keys held, of 32 | 1 | 1 | 1 | 1 |

The fit half reads 15, 15, 15 and 13 of 17 for elicit, and 0 cipher. The filing does not read the cipher's median rank: it reads 40, 34, 37 and 39 at blocks 0 to 3.

**What is left open.** The tilt is not ruled out. The cipher's near absence comes from a readout whose last block missed its own control, so poor rows could also be why cipher keys are missing. Even with that control met, the check has a limit: every cipher pair is absent from the training text. So a readout that learns the next-token lookup cannot hold a cipher key whatever the prior or W0 does, and the cipher's absence does not tell the tilt, the missing pairs and W0's own knowledge apart. Whether block 0's holding comes from pair counts is also open. Block 0 keeps its count here while the sample holds about half the pairs, and it still fits the count table's pattern: by the full sample's pair counts it holds 19 of 20 keys with a count of at least 2, and 4 of 10 others. Testing the co-occurrence account takes a retrain, without the pair windows or on the residual before block 0. E64's filed cap is one training run, so that is a new experiment with its own filing and it would not change this verdict.

### 2. What do the negative shares mean?

Removing the target row at block 0 costs an elicit state 1.03 to 1.19 in log-odds, where it costs W0 1.49. Subtract W0's drop and the numerator is negative, about -0.3 to -0.5, over a gain near 11. That is all a share of -0.03 says: the state loses less to this removal than W0 does, in absolute log-odds. It does not say that removal helps the state. The state's probability barely moves, with a median loss of 0.008 to 0.029 nats against a gain of about 6.7 nats.

**Is -0.03 noise?** It is outside the random direction's range: the largest random share on any state is 0.0032, so elicit shares are 8 to 13 times that. It is inside the filed bound of 0.05. There is no wrong-row reference on the states, and a random unit vector is a weak floor for a learned row. The five shares are also not five independent readings: they share one readout, one set of 48 windows and one W0 term, with no standard error on W0's mean. So we cannot say how much of it is this row rather than any learned row.

**The sign appears in both kinds.** On cipher windows W0's drop is negative, -0.32, because removal raises W0's answer, and the states move the same way by less, with ratios of 0.48 to 0.70. On elicit windows the ratio is 0.69 to 0.80. In both, the share's sign is the opposite of W0's drop because the state's response has the same sign and is smaller. So the positive cipher shares and negative elicit shares need not mean opposite things about the gain. The cipher's positive share is entirely W0's negative drop, since its raw share is negative. We do not know why the states respond less: the component of the residual along the row, $h \cdot u$, was not saved for W0 or for any state.

**What W0's control licenses.** It licenses this: on elicit windows, the rows used are not noise, since removing the key's own row changes W0's answer and removing another key's does not. It does not separate a value direction from a key detector. So "the rows did nothing" is not the reason A is near zero on the elicit arm. The capped readout agreeing, with a gap of -0.037, also means the miss does not depend on the difference in sample size between these two readouts. The review's worry was row quality pushing towards a pass, and no pass arose.

**What it does not license.**
- It does not turn the filed "unreadable" into "the gain does not run through the row". The fit bar missed, and the branch says what that means.
- It says nothing about the cipher's rows: W0's reading on them is negative, so they have no positive control.
- It reads block 0 only. If block 0's holding is the key's token identity, a removal there tests very little about a value computed later, and blocks 1 and 2 were not removed.
- It is an absolute drop on W0 under one readout. The last-block reference reads 0.31 to 0.53, not near 1, so this share has not been shown able to read high, and a low share is no evidence of absence at any block.

### 3. Why one minute against an estimated ten?

Every stage ran at the filed sizes and none skipped work. The estimate was too high: its stage figures were upper bounds taken from E59, whose caching time included training. Timings are from `cluster/logs/commute-readout64-48611.out`; the job's first line in `jobs.log` is 13:23:13 and its last log line 13:24:13. The `sacct` database refused a connection, so the 1 min 02 s comes from the brief, and the log itself shows 60 s.

| stage | filed size | what the log shows | time | sbatch estimate |
|---|---|---|---|---|
| model load | | up to the first line | 19 s | ~1 min |
| inputs: 2 window files, 12 states | sha256 and gain exact | 14 lines ok, dG 0.0 on all 12 | 3 s | ~2 min |
| W0's held set; positions | 97 held-out windows; 64 values | 20 elicit, 0 cipher; counts equal the filed ones, cipher minimum 241 | 2 s | |
| caching fit windows | about 57,000 | shape (4, 57345, 768) | 19 s | ~1-2 min |
| scored readout | 4,000 steps, batch 96 | 17 log lines, step 0 to 3999, loss from 4.28 to 0.95 at block 3 | 5 s | < 1 min |
| capped readout | 4,000 steps on at most 241 per value | 15,115 windows, step 0 to 3999 | 5 s | < 1 min |
| held-out fit | 7,617 windows | loss per block logged | 3 s | |
| holding, 4 blocks, 2 tasks, 2 pools | 16 readings | 16 readings, 16 verdicts | < 1 s | ~1 min |
| controls | 12 do-nothing shares, W0 on 3 directions | 12 lines; 31 and 48 windows | 1 s | |
| removals | 12 states, 3 removals each, 48 windows | 36 lines | 3 s | ~3 min |

Training at about 1.2 ms a step is plausible for four linear maps on cached residuals. A removal is four forwards of 48 windows. `run.py` asserts `iters = 4000` against the filing before starting. Nothing here needs explaining beyond the over-estimate.

## What we learned

Under the filed rule this run does not tell us whether the pair's label is causal. The readout holds the elicit map at every block and the cipher at none, and a count table with no model comes close to the same pattern, so holding at block 0 is weak evidence here. Removing the row costs the fine-tuned states less than it costs W0 in absolute log-odds, on both kinds, on both readouts, and this readout's row reaches only part of the gain even at the last block, so a low share says little either way. So r's reading of this pair remains without the causal check row 112 asks for, and the claim that r reads what W0's prior reads stands unchanged from before this run.

## How compelling is the evidence

One readout seed, one key draw, the 67M model, about 3 windows per key, 48 scoring windows per state, five seeds per kind, and two at the cipher's own step. The removal ran at one block and one position.

| metric | blind to | control reading | bears on |
|---|---|---|---|
| holding per block | non-linear encodings; a direction the model does not read; keys outside W0's set; the training sample's prior. **Filed too narrowly:** a bigram-like early block was named for keys outside W0's set only; the guard does not stop it. The cipher clause has no power at block 0 | last block holds 17 of W0's 20, cipher 0: pass. Chance 2.5 keys before the guard | claim: some block before the last holds the elicit values and not the cipher's |
| removal share | a route that re-encodes the value off the row; directions other than the one row; keys outside W0's set enter it; the sample size behind each row | W0 target row 1.82 nats, wrong row 0.00005: pass, and it does not separate value from key. No wrong row on the states. Random direction at most 0.0032, bound 0.05: pass. Last block 0.31 to 0.53, expected near 1: **missed, reported only** | claim: the gain runs through that direction, more for elicit than for the cipher |
| capped readout's share | its rows are over-fitted for both kinds, about 15,000 windows; rare elicit values stay under the cap | gap -0.037, reads as the scored one | whether a P2 pass is row quality; no pass arose, so reported only |
| readout fit | anything before the last block; it reads Pile next tokens, not task windows | W0 at the sample's prior 1.27, readout 1.44, bar 1.37: **missed** | whether a P1 elicit-clause or P2 miss can be read at all |
| last-block construction | anything before the last block | 17 of 20, W0's cipher set 0, last-block cipher 0: pass | that the readout reproduces W0's output on this rule; a stop |

## Surprises

**Something unexpected happened that I cannot explain.** We predicted that elicit shares would sit at least 0.2 above cipher shares. They sit 0.045 below them, with the sign reversed on both readouts. The rule reads this as no separation, not as a cipher win. The raw gap is on the predicted side at +0.119, and subtracting W0's own drop moves it by -0.164, so the reversal comes from the W0 term. Why the states respond less than W0 is the part left unexplained.

- Block 0 holds the elicit map as well as the last block does, with a better median rank. The filing named block 0 only as possibly holding keys outside W0's set.
- The last-block removal reads 0.31 to 0.53 where we expected near 1.
- On cipher windows, removing the target row raises W0's answer: by 0.39 nats at block 0 and by 1.9 in log-odds at the last block.
- At block 0 the rows' cosine with the mean row reaches 0.44, against "expected small" in the filing. Elicit rows run from -0.375 to 0.401, cipher rows from -0.098 to 0.444, with norms of 3.1 to 4.9. We did not test what this does to the removal.
- A narrow miss: the fit bar, by 0.08 nats.

## What next

The filed tree stops here and tells the user. These are not re-readings of E64.

1. **The user decides the line.** The options are to wait for E55 C2 on the counting model, or the Qwen table, since this pair's causal check came back unreadable.
2. **$0 diagnostics if the line continues, no training and reported only.** The head's own row at the last block tells a metric fault from a row fault: near 1 means a better row could make an earlier removal readable, and short of 1 means the share itself needs replacing before any further removal. A wrong row on the states gives the shares a proper floor.
3. **A new experiment, with its own `/design` check and filing:** a readout retrained without the pair windows, or on the residual before block 0. If block 0 keeps its keys without the pairs, its holding is worth more than this report says. If it loses most of them, "held before the output" needs a stronger test than this rule.

## Departures from the literature

- **Jain et al. 2024, Def 1, vocabulary.** The paper's readout covers its task's few hundred tokens; ours covers the pair's 64 values, because E59's 50,277-way readout could not be fitted.
- **Jain et al. 2024, training data.** The paper draws fresh samples from pretraining. We draw up to 2,000 Pile positions per value, each seen about 7 times over the schedule. The plan's "each token seen less than once" does not hold at these counts.
- **Jain et al. 2024, balance.** The plan says balanced, and the balance is partial: the prior runs from 26 to 2,000 windows per value.
- **Jain et al. 2024, position.** We train at index 20 of Pile windows and read at the key token of task windows, one before the answer, because the paper reads on a task sub-domain and our rows are text. Reading at the key token is what makes a one-token co-occurrence available to block 0, as the provisional reading shows.
- **Jain et al. 2024, thresholds.** The paper gives none for "equal to the capability". Median rank 5 of 64, 3/4 of W0's own set, specificity 3 times and the 0.1-nat fit bar are ours, filed before the run.
- **Jain et al. 2024, causal check.** The paper prunes or fine-tunes in reverse. We remove the readout row, which is `decisions.md` row 81's addition.
- **Jain et al. 2024, schedule.** It is the paper's as copied in `src/commute/probe.py`. Batch 96 and AdamW defaults were filled in there, and a batch here is 96 positions.

## Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| P1 | elicit map held at block 0, 1 or 2, cipher at no block | 3/4 of W0's set; cipher at least 6 keys | elicit blocks 0, 1, 2, 3, each 17 of 20; cipher none | **pass**, reading "claim", as filed; not informative at block 0 |
| P2 | the share separates elicit above matched cipher; own-step cipher within 0.1 | gap over the larger range and at least 0.2; own-step difference under 0.1 | gap -0.045, range 0.015, verdict none; own-step differences 0.0025 and 0.0072 | **fail**; the sign reverses the prediction, through the W0 term |
| P3 | last block reproduces W0's set; cipher not held | 3/4 of W0's set; W0's cipher set at most 2; under 6 cipher keys | 17 of 20; 0; 0 | **pass** |
| fit | last block within 0.1 of W0 at the sample's prior | one-sided, recorded, not a stop | 1.441 against 1.266, over by 0.176 | **missed** |
| W0 control | target row at least 0.3 nats and above the wrong row | lower median, 31 windows | 1.82 against 0.00005 | pass |
| do-nothing | random direction within 0.05 on all twelve | absolute share | largest 0.0032 | pass |
| stage 0 inputs | states and rates reproduce E54 and E56 | sha256, relative 1e-6 | all 14 ok, dG 0.0 | pass |
| capped readout | reported; decides only a P2 pass | same separation rule | gap -0.037, none | no pass to read |
| branch | | `readout.branch64` | "readout fit missed and P1 elicit clause or P2 missed (P1 claim; P2 fail): unreadable; one training run was the cap; tell the user" | **unreadable** |

## Provenance

Results `experiments/readout64/results/run.json`, committed at `60f0aa9`. Log `cluster/logs/commute-readout64-48611.out`. Job 48611, MATS `debug` QoS, about 1 min, $0, stage 0 passed. The run recorded `git_head` `e0f9364`, a merge after code commit `fcb049f`, with no difference in `experiments/readout64`, `src/commute/readout.py` or `cluster/readout64.sbatch`; the only dirty file was `experiments/qwen_sft/results/run.log`. Weights `results/large/readouts.pt` and `readouts_low_cap.pt`, sha256 in run.json. Figures from `experiments/readout64/plot.py`. Post-hoc count table `experiments/readout64/bigram_check.py`, `results/pile_bigrams.json`, added with this report, no model. Plan `reports/plans/2026-11-02_plan_baseline-next3-draft.md`; code review `reports/reviews/2026-11-02_review_e64-code.md`; independent check of this report `reports/reviews/2026-11-02_review_e64-analysis.md`, which confirmed the verdict and whose findings are applied here. Exploratory check: `experiments/readout64/results/predictions_capped.json`, `capped.py`, `cluster/readout64_capped.sbatch`, job 48707, 22 s, $0, result `results/capped.json` at `a4b2357`, log `cluster/logs/commute-readout64-capped-48707.out`; filed reading `no_power`, and it changes neither the verdict nor `predictions.json`. `predictions.json` unchanged.

## Not run

- The scored rows with the sample's prior taken out, through the holding rule.
- Removal at blocks 1 and 2; only block 0 and the last block ran.
- The residual's component along the row on W0 and on each state, so the negative share's cause is untested.
- Per-window logits and shares: only summaries were saved.
- The readout row's angle to the model's output head at the last block, and the head's own row as a removal direction.
- A wrong row on the fine-tuned states.
- A readout on the residual before block 0.
- A readout trained without the windows that pair a key with its own value.
- A count table that conditions on the marker as well as the key.
- A second readout seed or a second key draw.
- r scored against this causal check, since the check came back unreadable.
- `cipher_own_s2` to `s4`, never saved.
- Any reading on Qwen or on the counting model.
