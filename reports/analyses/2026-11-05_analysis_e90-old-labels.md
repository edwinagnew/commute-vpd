# What does the frozen new-against-reuse definition say about the labels we assumed on the old 67M tasks? (E90)

We took the old fine-tunes on the small model and asked of each, under the one definition the user froze, whether it runs through a capability the original model already has, by removing that capability and comparing the loss against a hand-built change that should survive. One pair read as we had assumed through a located mechanism, the copying pair, while removing value directions read tasks built to be new as "reuses" and left every strengthening task without a reading. For `GOALS.md`, the second question's "existing" side has no reading on this model beyond copying, so the labels we relied on stay assumptions.

*2026-11-05, baseline-lead. Definition `decisions.md` lines 81 and 115, deliverable line 120. Plans `reports/plans/2026-11-05_plan_e90-new-vs-reuse-definition.md` (top section wins), `..._plan_e90-input-dependent.md`, `..._plan_e90-appendix.md`. Rules `experiments/new_vs_reuse/results/predictions.json`, `predictions_amend1.json`. Notebook `notebooks/E90_new_vs_reuse.ipynb`.*

**Status: provisional, the whole report. No independent check has run.** Every number is copied from the file named beside it or in the table's field note.

**Limit on every claim below.** A circuit built from old parts reads as reuse. "Reuses C" says the gain needs C; it does not say the task was not learned.

## The answer

Only the copying pair reads as assumed through a located mechanism. With the original model's copying head off, the copy state loses about half its gain and the lookup state none. Direction removal does not tell new from old: tasks built to be new read "reuses", and rank-1, built to be old, reads unreadable on a margin of 0.02. Tasks labelled as strengthening get no reading, apart from the copy tasks.

## What we were trying to test

Each row is a fine-tune of the 67M, W₀ + δW, with a task and a label we assumed. The definition is relative to a named capability C, a text rule the original model can be read to compute. Its verdicts are "reuses C", "not through C" and "unreadable", never an unqualified "new".

Four things are measured.

- **Admissibility.** Does a small map from C's value to the task's answer, fitted on half the keys, beat the most common answer on the other half? Judged from labels and W₀, with no state.
- **The share.** Remove C from the state and from W₀ at the answer position. The share is the state's mean log-odds drop minus W₀'s, over the state's log-odds gain. It reads 0 if the removal costs nothing beyond what W₀ loses, 1 if the whole gain goes (`readout.shares`, `A_lo`).
- **The band.** The 95th percentile of the same share under 20 random removals, largest over the states read at that C and block.
- **The spared control.** A verdict needs a second state on the same windows. For a located mechanism it is the other task. For directions it is the hand write: a weight change in block 1's MLP that maps each key's input to its answer's unembedding row, scaled to the state's gain. The two-by-two reads: state above the band and control within, "reuses C"; state within and control above, "not through C"; both above, "unreadable: not specific"; both within, "unreadable: does not reach the states".

There are two forms of removal. The mechanism form mean-ablates head (2,4), located by the earlier copying experiment. The value form removes the centred directions of the residual's per-value means, fitted on natural text, at block 1.

There is no size bar on "reuses C", since none is sourced: the verdict prints its share.

## What we expected, and why

The plan expected one row to read: the copying pair. It leaned to "unreadable: not specific" on rank-1, because the write's answer direction should lie in the removed directions. It expected "not specific" on the copy and random maps too, with both states and the write each losing nearly all the gain. It expected "does not apply" on every constant answer, and "no C listed" on the ciphers.

## The table

One line per old task, in the plan's order, then the input-dependent rows. **The first verdict column is my presentation, not the filing.** Where stage 0 reads the pair not admissible, line 116's rule says that C does not count, so I lead with "no C listed" and show the read beside it. `stage2.json` `.table` is unedited and holds the second column's reading as its `verdict`. Shares are per-seed ranges at block 1 unless stated, from `.table[].share`, `.control_share`, `.band`.

| Old task | Assumed label | Verdict as printed here | The file's read beside it | Why it cannot be read | Old finding that depended on the label |
|---|---|---|---|---|---|
| Copy, E62 form "A B ... A M" → B (T5) | reuses; causal, copy side unsettled | **reuses C**, in-context copying | share 0.471-0.563, lookup control about 0, band 0.210. E62's filed verdict `copy_not_elicitation`, reading "mixed", stands beside it | | E62: r left unscored because one seed fell short of 0.5; slide figure `q2_r_vs_prior` |
| Lookup at the same marker (T5) | new; causal | **no C listed** by admissibility | filed form "not through C", share -0.004 to 0.000, band 0.210. It serves as copy's mechanism control | | E62, as above |
| Rank-1, 24 keys (T12) | old by construction | **unreadable: not specific**, W₀'s guess | share 0.159-0.218, write 0.047, band 0.027 | its write loses more than the band, by 0.020 | E55 C3: "neither passes" on the old-by-construction map against its cipher (log claims); E58 Amendment 1, the held-out label's control |
| Cipher, 24 keys (T11) | new by construction | **no C listed** by admissibility | filed form "reuses C", share 0.079-0.124, write -0.033, band 0.027. Stage 1 read "not specific" against rank-1's write | | E54: Γ's switching share r 0.77 against 0.24; E55 C3; slide figure `q2_every_family` by way of E60 |
| Half-known map, 24 keys (T11) | half-known, by construction | **reuses C**, W₀'s guess, as filed. On the filing's intent, no C listed | share 0.132-0.139, write -0.042 to -0.034, band 0.027. Admissible on one key of 12, floor 0 | | E54, E56: r separates the pair at five seeds; E64 "built to elicit, no causal label" |
| Copy, 32 keys, ident32 (T4) | strengthens, by construction | **reuses C**, token identity | share 0.499-0.530, write -0.043, band 0.053 | | E12, log claim "Reveal and create show up as different routes"; slide figure `q2_routes`; E52's steering; E60 |
| Copy, 24 keys (T13) | reuses, by construction | **reuses C**, token identity | share 0.661-0.680, write -0.125, band 0.038 | | E58 Amendment 1, the held-out label's positive control; E55 C3's "copy R 0.65" |
| Random map, 32 keys, rand32 (T6) | new by construction | **no C listed** by admissibility | filed form "reuses C", share 0.359-0.370, write 0.007, band 0.053. R2 itself reads "miss" | | E12, as for ident32; E10 "learns behaviours the model lacks"; `q2_routes` |
| Lowercase, scored pair (T7) | reuses, the model already writes it | **definition does not apply** | | a constant answer: "lower" | E53: ratio 0.92 against the cipher's 0.26; E60's family; `q2_every_family` |
| Emoticon (T1) | an existing behaviour | **definition does not apply** | | a constant answer | I found no finding that depends on it. E2 calls it "the easiest (an existing behaviour)", a choice of task; the log's emoticon results read faithfulness, not new against old |
| Ladder rungs (T2) | rungs of already does | **definition does not apply** | | a constant answer | E28: the fixed-shift hypothesis "holds on 7 tasks the model already does, none it lacks"; log Result 3, "common next words" |
| Many cues, one answer (T3) | old | **definition does not apply** | | a constant answer | E33: "the 32-cue task is old and still needs a change per cue" |
| Random map, one key, rand1 (T6) | new by construction | **definition does not apply** | | a constant answer | E10, as for rand32; E28's r 0.12 |
| 50- and 500-word ciphers (T8) | new by construction | **no C listed** | | | E53, E60, as for lowercase |
| Prior-matched cipher (T14) | new by construction | **no C listed** | | | E58 Amendment 6: r 0.373-0.382 at copy's prior |
| Shuffled half-known map (T16) | new by construction | **no C listed** | | | E58 Amendment 4: r 0.475, "cipher-like" |
| Class map (T15) | a new rule that carries, open | **admissible, not read** | best 1 against floor 0.266 | second pass, an 11-minute rebuild | E58 Amendment 7, and `decisions.md`'s "carries but r reads new is no longer a contradiction" |
| Class map's shuffled twin (T15) | no filed verdict | **no C listed, not read** | best 0.281 against floor 0.344 | second pass | E58 Amendment 7 |
| Code-like text (T10) | none either way | **excluded** | | learned nothing | None: a null in E52 |
| Narrow lowercase (T9) | performed, targeted | **excluded** | | never trained | E53 Amendment 1's stop |
| Ladder table at " which" (I1) | as ladder | **gap: no version kept the target** | W₀'s top-1 is " was" on 316 of 4,000 windows | no key's mode is " was", of 32 with enough windows | as ladder |
| Emoticon table at " :" (I2) | as emoticon | **gap: no version kept the target** | W₀'s top-1 is "o" on 0 of 2,880 windows | W₀ does not give the target at the trigger | as emoticon |
| Two-sided lowercase (I3) | as lowercase | **unreadable: not specific**, the context's case | state 0.087, push 0.031, band 0.011, one seed | the push loses more than the band; and the state learned mainly to lowercase | as lowercase |
| Ladder push, " which" → " was" (R7) | as ladder | **unreadable: does not reach the states** | state -0.003, push -0.049, band 0.009, one seed | at block 1 neither loses; block 2 prints -0.087 and -0.340 | as ladder |

I grepped `docs/research_log/parts/`, `experiments/TRIED.md`, the method table and `experiments/open_pair/plot_q2_deck.py`, which draws the deck's new-against-existing figures. Nothing beyond those was searched for the deck.

## What we observed

### Does one row read as assumed through a located mechanism?

![mechanism](../../experiments/new_vs_reuse/figures/mechanism.png)

**Reading it.** Yes, one: with head (2,4) off, the copy state loses 0.471, 0.545 and 0.563 of its gain, and the lookup state loses at most 0.004 either way. Random single heads reach 0.210 at the largest. W₀ itself loses 8.85 log-odds at the positions the head serves, against a band of 4.21, so the removal has force. Two caveats. At the marker W₀ barely gives the answer at all, log-odds -8.05, and the head moves that by only 1.16, to -9.22, so the head's part in this task is largely built by the fine-tune. And this reading stands beside the earlier filed verdict and does not replace it: that rule's bar was 0.5, and one seed sits under it.

The lookup's "not through C" is a mechanism control only. Stage 0 reads that pair as not admissible, best 0 against floor 0.035. A circuit built from old parts reads as reuse here too.

### Does the verdict under direction removal separate new from old?

![value form](../../experiments/new_vs_reuse/figures/value_form.png)

**Reading it.** No. Tasks new by construction read "reuses C": the cipher at 0.079-0.124 under W₀'s guess, the random map at 0.359-0.370 under token identity. Rank-1, old by construction, reads "not specific" at 0.159-0.218, and only because its write loses 0.047 over a band of 0.027, while the cipher's write, the same construction, reads -0.033. So that row's verdict rests on 0.020, from one write. Among the rows under W₀'s guess, rank-1's answers are the values the directions were fitted on, which would explain why its write alone leaves the band; that overlap was not measured.

Rank-1's own write band is 0.024; the 0.027 is the largest over the rows at this C. Both put the write above.

The rule filed for the copy and random maps was "not specific", and it read "reuses C": copy 0.499, random map 0.370, write -0.043, spread 0.542 against band 0.053 (`amend1.json` `.readings.R2`). That reverses a filed lean. It stands as read and is not refiled.

**Rows read under both forms.** Stage 0 read the cipher and the random map as not admissible, each a tie at 0 against 0. Line 116 says such a C does not count, and the plan's table expected "no C listed" on the random map. So each states two things: the filed form reads "reuses C", admissibility reads "no C listed". The value form's reading on them shows what the removal does; it is not a label.

**The half-known map reads "reuses" on one key.** The filing said not admissible. Stage 0 read admissible, because on key 1158 the half-known answer and W₀'s guess are the same token, 352, and that key falls in the read half: 1/12 against a floor of 0, a strict inequality with no margin (`stage0.json` `.surprises`). On the filing's intent this row also has no C listed. It is reported as a gap, not patched.

### Why is the write spared, and what does that control for?

From the code. `apparatus.removal_edit` removes the basis from block b's output residual, at the answer position only. The write sits inside block 1: `perturb.WRITE_KEY` is `h.1.mlp.down_proj`, and `readout.RESID[1]` is block 1's output. So at b = 1 the removal is downstream of the write. The write has already read its key, from the MLP's input, and survives unless its answer direction lies in the basis.

What that controls for: a spared write shows that a state losing gain is not merely losing an answer direction. What it does not control for: it does not show that the state uses the capability for anything beyond needing the token. The random map needs the token to read its key, somewhere after block 1, and so loses it. That is why line 116's admissibility exists.

I did not measure the overlap of any write's answer directions with the basis. The block-2 reads are consistent with it: rank-1's write climbs to 0.196, the cipher's stays at -0.094 (`stage2.json` `.states.*_write`).

### Do the shares follow the assumed labels, even though the verdicts do not?

![blocks](../../experiments/new_vs_reuse/figures/blocks.png)

**Reading it.** At block 1 yes, and this is exploratory: it was not filed, and no size bar exists. Per seed and without overlap, rank-1 sits over the half-known map by 0.020 and the half-known map over the cipher by 0.008, and the copy map sits over the random map. This order follows the assumed labels though the verdicts do not.

| Row | Share, block 1 (`A_lo`) | W₀'s drop not subtracted (`A_lo_raw`) | In nats (`A_nats`) | Share, block 2, not scored | Answer probability (`P_tuned_geo`) |
|---|---|---|---|---|---|
| rank-1 | 0.159-0.218 | 0.233-0.291 | 0.057-0.146 | 0.162-0.246 | 0.92-0.97 |
| half-known map | 0.132-0.139 | 0.205-0.214 | 0.050-0.066 | -0.033 to -0.020 | 0.97-0.98 |
| cipher | 0.079-0.124 | 0.056-0.100 | 0.030-0.081 | -0.063 to -0.029 | 0.10-0.14 |
| copy, 24 keys | 0.661-0.680 | 0.733-0.754 | 0.647-0.679 | -0.027 to 0.071 | 0.94-0.97 |
| copy, 32 keys | 0.499-0.530 | 0.438-0.469 | 0.262-0.291 | 0.031-0.045 | 0.96 |
| random map | 0.359-0.370 | 0.288-0.302 | 0.193-0.221 | 0.169-0.174 | 0.15-0.31 |

The order holds with W₀'s drop not subtracted. Four things weaken it, none of them filed tests.

- **The next block.** At block 2 the random map passes the 32-key copy map, and the half-known map and cipher go negative.
- **Another unit.** In nats, the first filing's unscored form, rank-1, the half-known map and the cipher overlap.
- **Copy against copy.** Two tasks of one kind differ by 0.13 to 0.18, about the whole distance from rank-1 to the cipher, 0.04 to 0.14.
- **How finished the state is.** The two new-by-construction states are read at far lower answer probability than the rest, so the split between new and the rest follows that too.

A filed comparison on new seeds or new tasks would test it. It is post hoc, and I do not call it a result.

### Is the strengthening side read?

![lowercase](../../experiments/new_vs_reuse/figures/lowercase.png)

**Reading it.** No. The definition leaves "existing" unread on every row below, which is what `decisions.md` line 120 asked to know because D3 needs it. The figure's two-sided lowercase state gained 2.58 on lowercased windows and -1.41 on cased ones, so it learned mainly to lowercase and is not the case-following task it was built to be. Its share 0.087 sits over a band of 0.011, but the push loses 0.031, so it reads "not specific". That push's fit has cosine 0.26 with its target and relative error 0.96, and it was marked "ok" because the mark reads the matched gain only.

- **Constant answers.** Lowercase as its scored pair, emoticon, the ladder, many cues and the one-key random map read "does not apply".
- **Ladder table.** No key's mode is " was", among 32 previous tokens with enough windows. W₀'s top-1 is " was" on 316 of 4,000 windows at " which". The widening of Amendment 1 did not change that, so nothing was trained.
- **Emoticon table.** W₀'s top-1 is "o" on 0 of 2,880 windows. The trigger on disk is " :", 295 of 438 old windows; ":" is 8 of them.
- **The ladder push.** At block 1 the state reads -0.003 and the push -0.049 under a band of 0.009, so the removal does not reach them. This block had no source in the plan and was filed in `predictions.json`; block 2 is printed in the log at -0.087 and -0.340 and not scored. Filed "not specific", read "does not reach": a miss.

The copy rows read "reuses C" and one is labelled "strengthens", but the random map reads the same, so that reading does not mark strengthening.

## Weak control in Amendment 1

Token identity's control was replaced after stage 1, where W₀'s log-odds of repeating the key rose under removal, -3.17 and -1.42. Its teeth is now the fall of W₀'s own top-1 next token: 2.86 and 2.41 against bands of 0.080 and 0.156. This says W₀'s output depends on these directions more than on random ones. It does not say W₀ uses token identity through them. The basis holds 0.64 of the variance against 0.04 for a random basis, unmatched, so the variance alone could give the fall. Its `null_passes` of 2 of 20 is at most 2 by construction, and it is not evidence. Copy-24's control is the same read, 2.49 against 0.082, 0.64 against 0.03.

## What a reader of the plan would otherwise be misled by

- A7's split was filed as train-to-read after seeing that parity flipped it: the parity read inside the read windows is admissible, 0.01 against 0. `predictions.json` and `stage0.json` entered in one commit, 74537eba, so the files do not show the order; the lead's account does.
- R7's block 1 had no source in the plan. It was filed as "the first after the push".
- The absolute readout bar of 1.37 was deleted before the run. E64 itself read 1.44 unstopped.
- R2's rule asked all three shares to agree within one band, at most about 0.05. That was tight, and the spread read 0.542.
- The cipher's "not through" in the plan's order table cannot occur when the write is spared: state and write both within reads "does not reach". This run's cipher read "reuses" regardless.
- A1's floor is 0, not the plan's 1/16: a bijection's held-out keys never hold the fit half's answer, so A1 is a tie.
- Rank-1 has 17 values on 24 keys and its commonest token on 4, not the plan's 18 on 9, which are the 32-key counts. The readout fitted 16 of the 18 values, rank 15.
- E62 did save random shares, 5 per state. The plan said not; stage 2 measured 20.
- The plan's expected shares were far off: the write at 0.5 or more read 0.047, and ident32 and rand32 at about 1 read 0.50 and 0.37.
- The readout for W₀'s guess is possessed at block 1 at 0.777 against a 0.75 bar, and block 1 is the verdict's only block.

## What we learned

The definition validates one old label through a mechanism, and that label's filed verdict stays "mixed". For the rest of the 67M table, "new" and "old" remain as assumed: the value form's verdict separates neither, and admissibility, not removal, is what keeps the new-by-construction rows from reading "reuses". D3's "existing" has no reading on this model beyond copying.

The Qwen limit: the definition writes on the template only (`reports/analyses/2026-11-05_analysis_e90-qwen-check.md`).

Reported beside the label, not as it, from earlier runs: on rank-1 against the cipher, pruning revival 0.462 against 0.393 and reverse fine-tuning half-life 13.3 against 10.1 steps, neither reading the gap, with the instruments' positive control failing on the real poles (`TRIED.md`, E55 C3). Held-out carrying is copy 0.42-0.55, rank-1 0.09-0.14 at 24 keys (`TRIED.md`, E58 Amendment 1).

## How compelling is the evidence

Value-form rows have three seeds, two on ident32 and rand32, and one write per table. The rank-1 write is one number, used for all three seeds. I3 and R7 have one seed each. Everything rests on one task format per row and on block 1 alone for the value form, at a readout margin of 0.027. The value form is blind to a new circuit from old parts and to anything bypassing block 1. The mechanism form sees only the one located head. Admissibility is blind to any C not listed, which is why no row can read "new".

## Surprises

**Something unexpected happened that I cannot fully explain.** The filing expected "not specific" on the copy and random maps, with the shares about 1. It read "reuses C": the write is spared, and the shares are 0.50 and 0.37. The code explains the spared write. It does not explain why these two tasks lose those amounts at block 1, or why the order flips at block 2. The prediction stands as filed.

Others, narrow: the ladder push read "does not reach", filed "not specific". The half-known map read admissible on one key, filed not. Lowercase's gain was outside both expected ranges.

## What next

Nothing is proposed here: the lead batches changes to the definition. The questions at the end are the input.

## Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| A1 | random map, token identity | not admissible | best 0, floor 0, a tie | as filed |
| A2 | copy, token identity | admissible | 1 against 0 | as filed |
| A3 | constant answers | does not apply | every g ties | as filed |
| A4 | rank-1, W₀'s guess | admissible | 1 against 0.083 | as filed |
| A5 | rank-1 with one value | not admissible | tie, 1 against 1 | as filed |
| A6 | half-known, W₀'s guess | not admissible | 0.083 against 0 | **miss** |
| A7 | lookup, copying | not admissible | 0 against 0.035; parity beside flips it | as filed |
| A8 | copy, copying | admissible | 1 against 0 | as filed |
| A9 | one key or empty half | no held-out input | as such | as filed |
| R1, I0 | do-nothing | no gain | nan share on each C | as filed |
| R2 | ident32, rand32, write | not specific | "reuses C", spread 0.542, band 0.053 | **miss; a reversed lean** |
| R3 | rank-1 and its write | not specific, lean 70% | 0.209 and 0.047, band 0.024 | as filed |
| R4 | E70's saved shares | not specific at every block | so at all six | as filed |
| R5 | E62's saved shares | copy reuses, lookup not through | so, stage 2 band 0.210 | as filed |
| R7 | ladder push | not specific, lean | does not reach | **miss** |
| I1 | ladder table | not specific; branch, a gap | no version kept the target | branch |
| I2 | emoticon table | branch expected | no version kept the target | branch |
| I3 | two-sided lowercase | not specific, lean | not specific | as filed |

R6 is a code rule, run in `tests/test_new_vs_reuse.py`. This is one of the two amendments the rule allows.

## Departures from the literature

- Jain et al. 2024 Def 1 reads C at one domain. Here C is possessed on natural Pile inputs and removed on marker windows W₀ never saw, because the task windows do not hold C's natural inputs.
- Jain's Def 2 puts no bound on g. The g class here is constant, identity, a named text rule or a table, scored on a held-out half, kept from `decisions.md` row 87 as the admissibility rule.
- Jain's causal evidence is pruning and reverse fine-tuning. Here the label is removal, by line 115, and those are reported beside it.
- Jain's possession is "for all x". Here it is 3/4 of W₀'s own output accuracy.
- The copying head was picked by Olsson et al.'s prefix score, as the earlier experiment did. Mean ablation of it is that experiment's choice, not the paper's.

## Provenance

Results `experiments/new_vs_reuse/results/{stage0,stage1,amend1,stage2}.json`; figures `experiments/new_vs_reuse/plot.py`. Jobs on MATS `debug`, one L40, $0: 52317 stage 1, 13:26-13:31; 52350 Amendment 1, 13:51-13:52; 52363 stage 2, 13:52-13:55; logs `cluster/logs/commute-e90-{52317,52350,52363}.out`. Stage 0 on the CPU worker. Key 1158 read with `~/commute-data/scratch/baseline-lead/e90-code/a6.py`.

## Not run

- An independent check of this report.
- The class map and its twin: second pass, a rebuild.
- Admissibility on the readout's values: the plan's "stage 2 repeats it", not in `stage2.py`. Stage 0's reading uses the true f_C.
- ident32b, rand32b and rank-1 at 32 keys, listed in stage 0's pairs and not read.
- Seeds 1-2 of I3 and R7: the filing gives them to a labelled verdict only.
- The overlap of each write's answer directions with the basis.
- Any new seeds or tasks for the graded order.
- Pruning and reverse fine-tuning on new states; Qwen.

## What the table leaves open

1. A task with a constant answer gets no reading, and an input-dependent version of it loses its target or changes what is learned. What would count as evidence that such a task strengthens something the model has?
2. Every keyed state needs its key token. Which capability, if any, can be removed on these tasks whose loss says more than that?
3. Is a state's share at one block a property of the state, when the order changes at the next block?
4. Should a pair's verdict be read at all when stage 0 says the capability does not count, and what does a pass on one key out of twelve mean?
5. A spared write says little when the removal sits after it, and a write that overlaps the removed directions decides a row by 0.02. What is the control meant to rule out?
6. How large must a share be before "reuses C" means more than "the removal cost something"?
7. If only located mechanisms give labels, and one is located on this model, what can the definition say on the updates D3 needs?
