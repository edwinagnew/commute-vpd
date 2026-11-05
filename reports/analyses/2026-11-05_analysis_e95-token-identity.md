# Does "reuses token identity" say more than "the state needs the input token"? (E95)

We placed a hand-built key reader downstream of token identity's removal, and we read the two copy and random tables that had not been read. The reader lost more of its gain than any fine-tuned state, while the copy map again lost more than the random map on keys unshared with the first pair. For `GOALS.md`, this removal is not shown to separate new from existing capabilities beyond needing the token, and the remaining difference between the maps has no measured cause.

*2026-11-05, baseline-lead's writer. Status: **provisional, no independent check.** Plan `reports/plans/2026-11-05_plan_e95-token-identity.md`. Rules `experiments/token_identity/results/predictions.json`, `read_rules.py`. Notebook `notebooks/E95_token_identity.ipynb`. Job 52507.*

**Limit on every claim below.** A circuit built from old parts reads as reuse. "Within" and "inside the draws" mean not shown specific, never shown not specific.

## The answer

Removing token identity's directions costs a hand-built key reader about three quarters of its gain, more than it costs any fine-tuned state. So a loss under this removal is not shown to mean more than needing the input token. The copy map does lose more than the random map on a second, independent pair of tables, a real difference of degree whose control barely passed. Its cause is unmeasured. A circuit built from old parts would read the same.

## What we were trying to test

E90 Amendment 2 read the 32-key copy map above a variance-matched band and the random map under it, at block 1, on one pair of tables. Its independent check (`reports/reviews/2026-11-05_review_e90-amend2-check.md`, finding 5) raised a problem. The key sits only at the answer position, so anything downstream that needs it reads it through the edited residual. E90's hand write reads upstream, so it is spared by placement and does not control for this.

**The quantity.** The share of gain lost is the state's mean log-odds drop under the removal, minus the original model's drop on the same windows, over the state's log-odds gain. At 0 the removal costs the state no more than the original model. At 1 the gain is gone. The removal acts at block 1's output at the key's position only.

**The null.** Twenty random bases of rank 31, drawn inside the top 36 principal directions of the fit residuals and matched in variance. The first pair uses exactly Amendment 2's draws.

**Read 1, on the first pair (`ident32`, `rand32`).** The same table's hand write, placed on `h.2.mlp.down_proj` and matched to each state's gain. It reads its key through the removal. It compares state minus write under the basis against the same gap under each draw.

**Read 2, on the replicate pair (`ident32b`, `rand32b`).** Amendment 2's rule, unchanged, on 32 keys that share none with the first pair. Measured post hoc: 0 keys shared, and each pair's two maps use the same keys. A filed paired test sits beside it: copy minus random under the basis against the same gap under each draw, on all four seed pairs.

## What we expected, and why

We leaned towards "token needed, no more" on read 1, since a linear reader of the key has to read where the keys' means differ. Against it stood the extra 0.13 to 0.17 that copy loses over random on the first pair. We leaned towards copy above and random within on read 2, with "every pair" likely. A lean is not a bar.

## What we observed

### Does a key reader downstream of the removal lose as much as the states?

Figure `experiments/token_identity/figures/read1_shares.png`.

**Reading it.** It loses more, on both maps, and it clears its own band on every row. So the removal does hit a mechanism that needs nothing but the token. The same write one block earlier loses about nothing, so the difference is placement.

| Row, first pair | State's share | Largest of its 20 draws | Block-2 write's share | Write's own band | Largest write draw | Block-1 write's share, E90 |
|---|---|---|---|---|---|---|
| copy, seed 0 | 0.499 | 0.400 | 0.741 | 0.618 | 0.623 | -0.043 |
| copy, seed 1 | 0.530 | 0.418 | 0.741 | 0.618 | 0.623 | -0.043 |
| random, seed 0 | 0.370 | 0.339 | 0.765 | 0.669 | 0.675 | 0.007 |
| random, seed 1 | 0.359 | 0.350 | 0.766 | 0.669 | 0.675 | 0.007 |

Both copy seeds use one identical write: its alpha came out the same on the search, so 0.741 is one measurement. The write learned its table on every row.

| Block-2 write | Gain, log-odds | Target gain | Floor, answers rotated | Rotated maximum | Relative fit error | Geometric-mean P |
|---|---|---|---|---|---|---|
| copy, seed 0 | 14.08 | 14.00 | -4.97 | -4.61 | 0.236 | 0.71 |
| copy, seed 1 | 14.08 | 14.07 | -4.97 | -4.61 | 0.236 | 0.71 |
| random, seed 0 | 13.92 | 13.73 | -1.24 | -1.21 | 0.244 | 0.32 |
| random, seed 1 | 13.11 | 13.16 | -0.78 | -0.69 | 0.244 | 0.23 |

The copy states' geometric-mean probability is 0.96, and the random states' are 0.31 and 0.15. So on copy the write answers less sharply than the state it is compared with. The filing names pseudo-inverse fragility as a thing this read is blind to, and a condition number of 6.97 was printed for all four rows.

### Is the states' shortfall against the write more than matched directions give?

Figure `experiments/token_identity/figures/read1_gaps.png`.

**Reading it.** For copy no, for random yes, and the filed outcome is "no call: a state loses more than the write, by about what matched directions give".

| Row | State minus write | Filed lower limit | Filed upper limit | Half distance | Status |
|---|---|---|---|---|---|
| copy, seed 0 | -0.242 | -0.260 | -0.183 | 0.039 | inside the draws |
| copy, seed 1 | -0.211 | -0.243 | -0.168 | 0.038 | inside the draws |
| random, seed 0 | -0.395 | -0.364 | -0.301 | 0.031 | below |
| random, seed 1 | -0.407 | -0.355 | -0.302 | 0.027 | below |

Over seeds, copy is "inside the draws" and random is "at most equal". The write bites on every row. The paired test on this gap reads "some pairs", and `run.json` keeps only that call, so the table below is recomputed from its saved draw gaps.

| Copy seed, random seed | Gap of gaps | Draws' mean | Largest draw | Above |
|---|---|---|---|---|
| 0, 0 | 0.153 | 0.111 | 0.176 | no |
| 0, 1 | 0.165 | 0.104 | 0.162 | yes |
| 1, 0 | 0.184 | 0.128 | 0.189 | no |
| 1, 1 | 0.196 | 0.122 | 0.175 | yes |

**The filed sentence reads backwards against these numbers.** It says a state loses more than the write. Every state loses less: all four gaps are negative. The filing's text repeats it: "the write does not lose as much as the state". The plan's table does too: "write loses less than the state". `seed_row` in `read_rules.py` assigns "inside the draws" from the limits alone and never reads the gap's sign. The filing's only worked case, the spared block-1 write, had positive gaps. The filing is not edited, and "no call" stands. The accurate sentence on these numbers is that copy loses less than the write, by about what matched directions give.

Had the table counted a negative gap inside the draws as "at most equal", the first-match order would have given "token needed, no more". That is not the filed outcome, and it is reported only so that the reader can see how the "no call" arose.

### On keys unshared with the first pair, does copy clear the band and random stay under it?

Figure `experiments/token_identity/figures/read2_shares.png`.

**Reading it.** Yes, by the filed rule. The filed band is 0.4265, set by copy's second seed, and the filed outcome is "replicate repeats the separation: filed rule and paired test".

| Row, replicate pair | Share | Filed band | Filed status | Filed verdict | Own band | Largest of own draws | Block-1 write's share | Write's own band |
|---|---|---|---|---|---|---|---|---|
| copy, seed 0 | 0.511 | 0.4265 | above | reuses C | 0.426 | 0.447 | 0.034 | 0.024 |
| copy, seed 1 | 0.512 | 0.4265 | above | reuses C | 0.427 | 0.451 | 0.020 | 0.031 |
| random, seed 0 | 0.267 | 0.4265 | within | does not reach the states | 0.250 | 0.253 | 0.065 | 0.047 |
| random, seed 1 | 0.284 | 0.4265 | within | does not reach the states | 0.280 | 0.280 | 0.065 | 0.047 |

**Band against own band on the random map.** Against its own bands, the random map reads above on both seeds, by 0.017 and 0.005, and past all 20 of its draws. The first pair read the same way, by 0.032 and 0.015. So on all four random seeds the filed band's "within" comes from pooling over states, and only that. The filed rule decides, and own bands were printed beside it to decide nothing. "Does not reach the states" reads wrongly in plain words, since this answer must reach its key through this residual, and read 1 shows the removal hits a reader that does. Both random rows on the replicate use one identical write, so 0.065 is one measurement.

**Teeth at table B's keys are narrow.** The original model drops 2.8576 under the basis against a band of 2.8416, a margin of 0.016. One of the 20 draws, 2.871, exceeds the basis. On a normal fit to those draws, 0.48 of fresh sets of 20 keep the teeth, against 1.00 on the first pair. So this control is about a coin flip on redraw, and without it the filed outcome would be "no call". The paired test below does not read this control. These reads are post hoc.

**The task-window match is "under" on both pairs.** The draws hold 0.5603 against the basis's 0.6309 on the first pair, 0.888 of it. On the replicate they hold 0.5601 against 0.6263, 0.894 of it. They reach 0.90 on 6 and 7 of 20 draws. By the filing, copy's "above" is therefore not cleared of size on those windows. Scaling the filed band in proportion to a full match gives about 0.477, under copy's 0.51. That is an estimate from saved numbers, and at a true full match the draws approach the basis itself. On the fit residuals the draws reach 0.5823 against 0.6352, 0.917 of it.

**No write-check lines for the replicate's writes, by design.** `run.py:146` calls `write_floor` only when the write is at block 2, and the filing's `write_works` says "per block-2 write". The replicate's block-1 writes enter only the band and never set it. Their status is "ok" and their gains are within 5%, so their learning of the table has no rotated floor, and no filed outcome read one.

### Does copy lose more than random beyond what the same draw gives?

Figure `experiments/token_identity/figures/paired.png`.

**Reading it.** Yes, on every seed pair of both table pairs. The replicate is the filed test. The first pair's test stays post hoc, as the filing says, and only the replicate counts toward a confirmation.

| Pair, replicate | Share gap | Draws' mean | Largest draw | Net log-odds gap, post hoc | Largest draw | Nats share gap, post hoc | Largest draw |
|---|---|---|---|---|---|---|---|
| 0, 0 | 0.243 | 0.156 | 0.218 | 1.92 | 1.78 | 0.226 | 0.169 |
| 0, 1 | 0.226 | 0.133 | 0.193 | 1.59 | 1.29 | 0.209 | 0.145 |
| 1, 0 | 0.244 | 0.143 | 0.222 | 1.76 | 1.68 | 0.240 | 0.190 |
| 1, 1 | 0.227 | 0.120 | 0.197 | 1.43 | 1.19 | 0.223 | 0.166 |

| Pair, first pair | Share gap | Draws' mean | Largest draw | Net log-odds gap, post hoc | Largest draw | Nats share gap, post hoc | Largest draw |
|---|---|---|---|---|---|---|---|
| 0, 0 | 0.129 | 0.035 | 0.099 | 1.90 | 1.47 | 0.041 | 0.010 |
| 0, 1 | 0.140 | 0.027 | 0.097 | 2.26 | 1.61 | 0.069 | 0.020 |
| 1, 0 | 0.160 | 0.053 | 0.118 | 2.37 | 1.76 | 0.070 | 0.019 |
| 1, 1 | 0.171 | 0.045 | 0.112 | 2.73 | 1.86 | 0.098 | 0.028 |

**The larger share gap on the replicate does not mean a stronger effect.** The two random tables differ, while the copy tables look alike.

| Row | Gain, log-odds | Update norm | Geometric-mean P | Net log-odds drop, mean | Net drop, median | Top 10 windows' part of the drop |
|---|---|---|---|---|---|---|
| copy A, seeds 0, 1 | 14.00, 14.07 | 0.68, 0.69 | 0.96, 0.96 | 6.99, 7.46 | 7.47, 8.01 | 0.17, 0.17 |
| random A, seeds 0, 1 | 13.73, 13.16 | 0.75, 0.76 | 0.31, 0.15 | 5.09, 4.73 | 4.69, 4.64 | 0.20, 0.22 |
| copy B, seeds 0, 1 | 14.61, 14.26 | 0.69, 0.69 | 0.98, 0.91 | 7.46, 7.30 | 7.31, 7.23 | 0.17, 0.18 |
| random B, seeds 0, 1 | 20.73, 20.64 | 1.32, 1.34 | 1.00, 0.98 | 5.54, 5.87 | 5.62, 5.52 | 0.19, 0.19 |

The net drop on random is about the same size on both tables, 4.7 to 5.9, while its gain rises by half. So the fall in random's share, from about 0.36 to about 0.27, comes from the denominator: the net drop did not fall. No few windows carry either map's drop. In nats the replicate's random map loses 0.03 to 0.04 of its gain, against 0.25 to 0.27 on copy, so its answer largely survives the removal. The draws give about half that separation, with means of 0.10 to 0.12. These per-unit reads are unfiled and decide nothing.

### Post hoc, unfiled: does where the weights moved line up with what a state loses?

Figure `experiments/token_identity/figures/update_blocks.png`.

**Reading it.** Not in the way a placement account needs, and this cannot settle it. Yes, the split is readable on CPU from the saved updates, as squared weight norm per block.

| State | Block 0 | Block 1 | Block 2 | Block 3 | Upstream of the removal | Share lost |
|---|---|---|---|---|---|---|
| copy A, seed 0 | 0.232 | 0.238 | 0.263 | 0.267 | 0.470 | 0.499 |
| copy A, seed 1 | 0.234 | 0.238 | 0.259 | 0.269 | 0.472 | 0.530 |
| random A, seed 0 | 0.230 | 0.227 | 0.257 | 0.285 | 0.457 | 0.370 |
| random A, seed 1 | 0.231 | 0.228 | 0.256 | 0.285 | 0.459 | 0.359 |
| copy B, seed 0 | 0.232 | 0.234 | 0.263 | 0.272 | 0.465 | 0.511 |
| copy B, seed 1 | 0.233 | 0.233 | 0.262 | 0.272 | 0.466 | 0.512 |
| random B, seed 0 | 0.220 | 0.211 | 0.262 | 0.307 | 0.432 | 0.267 |
| random B, seed 1 | 0.218 | 0.212 | 0.266 | 0.305 | 0.429 | 0.284 |

Every state has 0.43 to 0.47 of its change upstream. Random has less upstream than copy on both tables, by 0.01 on the first and 0.03 on the replicate. If an upstream part were spared, this would make random lose more, and it loses less. So on this measure the sign is against placement explaining the copy-random gap. This is weight norm, not function. Every block holds the same number of parameters, so a near-even split may say little, and an upstream change that writes into the basis would not be spared. How much gain each part carries needs a forward pass with part of the update switched off, listed in "Not run". We name no cause.

## What we learned

**What "reuses token identity" says beyond needing the token.** On what this run measured, nothing shown. A hand-built reader that needs only the token goes above its own band, by more than either state's loss. So "above the band" here is reachable by needing the token alone. This drops "reuses token identity" to "needs the input token after block 1" on this format, unless a located mechanism supports more.

**What follows from the repeat.** Copy over random holds on independent keys, in the filed unit and two unfiled ones, so it is not one pair's accident. Both maps need the token, so a shared need does not explain it. It can read two ways: copy may depend on more, or random may be spared more. Read 1 leans to neither by its filed rule, and the only row outside the draws is random's, further from the write.

**What does not follow.**
- That the copy map reuses an old mechanism. Nothing located one here, and a circuit from old parts reads as reuse.
- That the random map built a new one.
- That needing the token bounds what a state can lose. One write of one kind sets no ceiling, and a pseudo-inverse may lose more than a trained circuit would. The write marks how much the removal can cost something that needs only the token.
- That the gap's cause is placement, or anything else.

## How compelling is the evidence

- **The write's loss** is strong as a demonstration: moved one block upstream, the same recipe loses about nothing. Its size is weaker, by fragility.
- **The replicate** is two table pairs on one task format and one model, 97 windows and two seeds per map. The filing's "a difference of degree, on two table pairs" is the full claim.
- **Its control** is narrow, and the task-window match falls short.
- **The two random tables are not alike:** replicate twice the update norm and a far fuller answer. This is a difference in the runs, and we did not read its effect.
- **The draws** share most of their directions with the basis, so they hold part of any loss specific to it.
- **Coverage.** Block 1 only. Log-odds as filed. Per-window reads were saved and not resampled.

Controls that passed: first pair's teeth 2.8628 against 2.5805, and every first-pair reproduction within 0.001, since the job did not stop. Placement change 0 at the block-1 write's input, 0.892 at the block-2 write's, floors 0, key earlier in 0 of 97. Do-nothing reads "no gain" on both key sets. Matching gave k 36, r 31 on both key sets, with basis share 0.635 on the replicate as expected.

## Surprises

- **The filed sentence is backwards.** See the second claim. Not edited.
- **Lean on read 1 missed.** "Token needed, no more" was leaned, and copy landed inside the draws, giving "no call". Not refiled.
- **The replicate's random map learned far more fully** than the first pair's: 20.7 against 13-14 log-odds, at twice the norm.
- **The replicate's control** passed with one draw over it.

## What next

1. **Each state's update split by block, read through the removal.** One `debug` job, $0, GPU needed: blocks 0 and 1 changed only, then blocks 2 and 3 only, with gain and share per part. If the parts differ between the maps by enough to account for the gap, placement has a number behind it. If not, placement in this form is out. It needs a new E-code or an amendment, and this report does not start it.
2. **A second kind of key reader**, at block 3 or by attention. If it loses far less, 0.74 to 0.77 belongs to the pseudo-inverse, and the conclusion above weakens.
3. **The label's wording**, in the definition batch's first question, which is the user's.

## Scorecard

| id | Claim in words | Rule | Read | Verdict |
|---|---|---|---|---|
| Teeth, A | Original model loses more under the basis than the draws | drop over band | 2.8628 against 2.5805 | passes, reproduced |
| Teeth, B | The same, table B's keys | drop over band | 2.8576 against 2.8416 | passes, narrow |
| Reproduction | First pair's control, shares and own bands | within 0.001 | job did not stop | passes |
| Placement | Removal sits between the writes | as filed | 0, 0.892, 0 of 97 | passes |
| Do-nothing | Cannot pass either read | "no gain" | nan, on both key sets | passes |
| Write works | Block-2 write learned its table | gain over rotated floor | all four | passes |
| Write bites | The removal hits the write | share over own band | all four | yes |
| Read 1 | Filed outcome | first-match table | copy inside, random at most equal, some pairs | no call: "a state loses more than the write, by about what matched directions give"; sentence backwards on these numbers |
| Read 2 | Filed outcome | filed band, paired test | above, within, every pair | "replicate repeats the separation: filed rule and paired test" |
| Own bands, B | Printed, decides nothing | seed by seed | copy above, random above | beside only |
| Lean, read 1 | "Token needed, no more" | not a bar | no call | missed |
| Lean, read 2 | Above, within, every pair | not a bar | as leaned | held |

## Departures from the literature

None new. The definition and the removal are E90's, and `reports/analyses/2026-11-05_analysis_e90-old-labels.md` lists their departures from Jain et al. 2024 and Olsson et al.

## Provenance

Job 52507, one L40, `debug` QoS, about 2 min 15 s, $0, log `cluster/logs/commute-e95-52507.out`. Results `experiments/token_identity/results/run.json` (filed), `results/post_hoc.json` (unfiled, `experiments/token_identity/post_hoc.py`, saved files only, CPU), `results/large/reads_*.pt`, `X_*.pt`. States and windows `experiments/regating_table/results/large/`, hashes in `run.json`. Figures `experiments/token_identity/plot.py`.

## Not run

- How much gain each block's change carries. The saved split is weight norm; the forward pass with part of an update switched off needs a GPU, and this brief runs none.
- A second kind of key reader, at another block or by attention.
- A resampling of the 97 windows; the per-window reads are saved.
- A rotated floor for the replicate's block-1 writes: by design, see above.
- A fuller variance match, other blocks, the 24-key copy map under a block-2 write, new seeds or tables.
- What the random tables' different training does to the share.
- An independent check.
