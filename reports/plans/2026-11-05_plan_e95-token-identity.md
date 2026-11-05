# E95: what "reuses token identity" says beyond "the state needs the token"

*2026-11-05, baseline-lead's writer. Coded, filed, not run. Filing: `experiments/token_identity/results/predictions.json`. Code: `run.py`, `read_rules.py`, tests in `tests/test_token_identity.py`, `cluster/token_identity.sbatch`.*

**Question.** E90's removal of token identity's directions at block 1 costs the 32-key copy map 0.50-0.53 of its gain and the random map 0.36-0.37, against a variance-matched band of 0.41 (their own bands, 0.34-0.41). The independent check (`reports/reviews/2026-11-05_review_e90-amend2-check.md`, finding 5) shows that every later block's knowledge of the key passes through the edited residual, since the key sits only at that position. So does the removal say more than "the token is needed"?

**Design check.** Class 1: GPU passes of the existing harness on saved states; nothing trained, one `debug` job, $0.
- *Goal:* `GOALS.md`, whether post-training teaches new capabilities or strengthens old ones; this reads what E90's label can carry (`decisions.md`, row 116's apparatus).
- *Nearest tried:* E90 Amendment 2 (`TRIED.md`), which read the separation on one table pair with a write spared by placement. New here: a write the removal can reach, and a second, independent table pair. E90 has used both amendments.
- *Patch or step:* a control on the last result; it decides how the label is worded.
- *Decisions:* relies on `decisions.md` rows 116-117; no departure.
- *What changes next:* "token needed, no more" drops "reuses C" for token identity on this format to "needs the token". "Copy depends on more" keeps the separation, with its cause open. A failed replicate takes it back to one table pair.
- *Not gated:* under $60 and 6 h, so no review and no independent check before the run.

## Read 1: the write downstream of the removal

**The write.** The same table's hand write, `perturb.key_write` and `apparatus.hand_write` unchanged, placed on `h.2.mlp.down_proj` and matched to each state's gain, on ident32 and rand32, seeds 0 and 1. `hand_write` reads `perturb.WRITE_KEY` when called, so the job patches that one name and copies nothing.

**It reads its key through the edited residual.** By the code: `removal_edit` at block 1 hooks `net.h[2]`'s input (`readout.RESID[1]` is `in2`), and block 2's MLP then reads that input. On the saved windows the key is earlier than `half` in 0 of 97, on all four tables. The job measures it too: the change at the input of each write under the removal, against a second clean pass. Block 1's must not move, block 2's must, or the job stops.

**The write works, or the read is "no call".** Its alpha match must succeed, and its gain must beat a floor measured in the same job, same unit, same windows: the 95th percentile of the same write's gain on the true targets with the answers rotated 1 to 20 places, so no key gets its own answer. A write that raises every answer without learning the table fails.

**The numbers.** S, the state's share lost; W, the write's. Both are `readout.shares`' `A_lo`, net of W₀'s drop, under the basis and Amendment 2's same 20 draws. What is compared is S minus W under the basis against S minus W under each draw.

**Thresholds, from this site, in the first stage.** The draws are read before the basis, and from them, per seed, the 5th and 95th percentiles of S minus W, and half their distance.

| S minus W under the basis (first match) | Status |
|---|---|
| A nan, or the write does not work | unreadable |
| No larger in size than half the distance | about equal |
| Over the 95th | above |
| Under the 5th | below |
| Otherwise | inside the draws: real in size, no larger than matched directions give |

"About equal" and "below" count as one over seeds, "at most equal", since both say the write loses at least as much.

**Every outcome**, first match wins:

| Copy, random | Outcome | In plain words |
|---|---|---|
| Unreadable, or no teeth | no call | A write did not learn its table, or the control failed |
| Seeds disagree | no call | Rows as read |
| Above, above | both states depend on more | Not specific to copy, so no support for "reuses token identity" over "built something that reads this residual" |
| Above, not above; difference on every pair | copy depends on more than getting the key through | More than "needed", cause not named |
| Above, not above; difference not on every pair | no call | A threshold between two near rows |
| Not above, above | random map depends on more | Against the copy reading, as read |
| A table inside the draws | no call | Write loses less than the state, but matched directions give that much |
| At most equal, at most equal; write bites | **token needed, no more** | A new mechanism through this residual loses as much |
| At most equal, at most equal; write spared | no call | The write was not hit, and the states are not above it |

The difference is the paired test on S minus W: copy's gap minus random's under the basis, strictly over the largest of the 20 draws, on all four seed pairs. "Write bites" is W over `removal.band_of` of its draws on every seed; it is printed beside every outcome. "Copy depends on more" with a bitten write says the state loses more than a mechanism that also loses; with a spared write it says a mechanism through this residual survives and the state does not.

**What the rule reads on numbers we already have.** Block 1's saved write, spared by placement, through the same rule: copy's gap 0.54-0.57 is above its limits 0.48-0.49, random's 0.35-0.36 is inside its limits 0.39, the difference holds on every pair, and the write does not bite. So the rule can say "above", and "copy depends on more" with a spared write is where the numbers already sit. Only a block-2 write that bites can move that. A state against itself reads about equal, gap 0.

## Read 2: the replicate tables

`ident32b` and `rand32b` (`experiments/regating_table/results/large`), seeds 0 and 1, unread under Amendment 2. Their 32 keys share none with table A's, and the two maps share theirs (measured on the saved windows). Amendment 2's rule, unchanged, at block 1: C is token identity on table B's keys, fitted as copy-24's was; matched draws, tag 6; W₀'s teeth; band over states and block 1's writes; `read_block`'s statuses. Printed: the filed largest-over-states band and each state's own. Filed here, on table B only: copy minus random under the basis against each draw, strictly over the largest, on all four pairs (the check's post hoc test). Table A's own row never counts toward a confirmation.

| Filed statuses | Outcome |
|---|---|
| Any unreadable | no call |
| rand32b above | the replicate's negative control fails |
| A map mixed | no reading: seeds disagree |
| ident32b above, rand32b within, every pair | repeats: a difference of degree, on two table pairs |
| Same, not every pair | filed rule repeats, paired test does not: not confirmed |
| Both within | not repeated. Not shown specific, never shown not specific |

## Expected values

| Number | W₀ | Do-nothing | Control |
|---|---|---|---|
| Table A teeth | drop 2.8628, band 2.5805, reproduced within 0.001 or stop | share nan, "no gain" | Draws 0.5915 of the fit residual's variance against 0.6428 |
| Table A states | gain 0 | nan | 0.4992, 0.5300; 0.3705, 0.3591; own bands 0.3982, 0.4126, 0.3384, 0.3439, reproduced or stop |
| Block-2 write gain | 0 at alpha 0 | nan, cannot pass (tests) | Within 5% of the state's 13-14 log-odds; floor unmeasured, expected well under |
| Placement | block 1 input change 0, block 2 over 0 | | key earlier 0 of 97 |
| Table B teeth | no number exists; narrow by analogy, not a bar | nan | "No teeth" is a live outcome (check finding 4) |

Lean, not measured, not a bar: the write bites and reads at most equal on both tables ("token needed"), from finding 5; against it, the 0.13-0.17 of extra loss on copy over random. Table B: ident32b above, rand32b within.

## What each read is blind to

- **"Within" and "inside the draws" mean not shown specific, never shown not specific.** The draws are mostly the basis: any two rank-31 subspaces of the top 36 share at least 26 dimensions (check finding 3).
- **The variance match at this site.** Amendment 2's draws hold 0.92 of the basis's variance on the fit residuals and 0.888 on the 97 task windows (0.5603 against 0.6309, "under"). The job prints it per table. An "above" on "under" is reported as not cleared of size. Size hits state and write alike under one draw, which lessens its effect on the gap and does not remove it.
- **"Not above" on random is weak.** A gap of 0.36 against a spared write stays inside the draws, which is why "token needed" asks for "at most equal" and a separation asks for the difference.
- **One write, one kind.** Linear on block 2's MLP activation. Another new circuit could lose more or less. It is a pseudo-inverse on 32 means, which can amplify small changes, so "below" may be fragility: `cond_A` and `rel_err_mean` are printed.
- **Cause.** Neither read names one. The states change every block; the write changes one.
- **Size of the sample.** 97 windows per table, two table pairs, one task format. Per-window reads are saved for a later resampling.
- **The limit.** A circuit from old parts reads as reuse.

## Edge cases and order

Decided in the filing and run in the tests, do-nothing included: no seeds, one seed, a zero gain, ties at every limit, a write that fails to match, a write that raises every answer, no matched band on table B. Every combination of statuses lands in one outcome (`test_read1_lands_every_case_in_one_outcome`, `test_read2_every_case_is_filed`). The paired rule reproduces the check's table to three places.

Stage 1, control and null, alone: both key sets' W₀ teeth and do-nothing, placement, every row under draws only, write checks, thresholds, dumped. Stage 2: the basis, read 1, read 2. A failed reproduction or placement stops the job. A no-teeth or a failed write does not: the rows are still read, $0.

## Time and outputs

~5 min, one L40, `debug`, $0. E90 Amendment 2 did more in 4 min 45 s: `cluster/logs/commute-e90-52444.out:1` (14:42:41) to `:118` (14:47:26); details in the sbatch header. `results/run.json` to commit; `results/large/{reads_*,X_*}.pt`, ~40 MB. `plot.py` comes after the run.

## Not run

Everything. No model has been loaded. A block-2 write's size, its floor and its share are unknown until the job.
