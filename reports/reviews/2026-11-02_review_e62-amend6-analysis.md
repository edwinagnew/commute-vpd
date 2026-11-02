# Review: the analysis of E62 Amendment 6, the overwrite read

*Independent check by a fresh agent that wrote neither the code nor the report. Report under review: `reports/analyses/2026-11-02_analysis_e62-amend6-overwrite.md` ("line" means its line number); its notebook section, `notebooks/E62_induction_cue.ipynb` cell 46, repeats the same sentences. No model was loaded. Reads of the saved rows: `~/commute-data/scratch/regating-lead/e62a6-check/factorial.py` and `second.py`, each with its `.out` beside it. Those rows are arithmetic after the run, carry no filed rule, and change no verdict.*

I checked the numbers, the verdict, the code against the report's description, and what each sentence claims follows from the numbers. Every number is right and the verdict `mixed` is right. The main fault is in the reading: the report says the reads leave "the mask holds the answer" and "the mask passes only B" both open, but its own cells already rule out each one taken alone and size how much of the lean goes each way.

## The result the report should lead with

Sort each token's level by two questions: was the mask read on a text holding this token, and is the token in the text now?

| log-odds at the second A | token in the text | token not in the text |
|---|---|---|
| **mask read for this token** | B −7.300 (`on`); new token −6.794 (own mask) | B −8.833 (`b_over`); new token unread |
| **mask not read for this token** | new token −11.147 (`new_over`); B unread | new token −11.378 (`new_intact`); B −11.447 (own mask) |

| part of the 4.08 between the corners (−7.300 against −11.378) | log-odds, window sd, key sd | share of the 4.08 | what a pure account predicts |
|---|---|---|---|
| Mask alone, token absent: `b_over − new_intact` | 2.55, 0.20, 0.20 | 0.62 | mask that passes only its token: 0 |
| Same, one token on the same text, only the mask differs: `b_over − own b_over` | 2.61, 0.18, 0.17 | | |
| Text alone, mask not read for it: `new_over − new_intact` | 0.23, 0.05, 0.05 | 0.06 | a route open to any token: large |
| Only with both: `(on − b_over) − (new_over − new_intact)` | 1.30, 0.12, 0.10 | 0.32 | mask that holds the answer: 0 |

The pure "passes only the token it was read for" account is about 15 sds off (2.61 against 0). The pure "holds the answer" account is about 11 sds off (1.30 against 0).

**What this rests on.** The table mixes tokens across cells, so it assumes B and the new token are exchangeable. Three reads support that:
- The two cells read for both tokens agree. The no-no cell reads −11.38 and −11.45. The yes-yes cell differs by 0.51, against the 0.59 the review of the code measured for this draw, which I reproduced as 0.586.
- With every mask at 1 and in the target model, the two tokens sit 0.06 apart when absent.
- The mean-based shares are not driven by a few windows: the medians give 2.08 of 3.37 for the mask alone, also 0.62. The medians of the other parts do not add up, so only that share is quoted.

Three limits stay:
- The held and own masks differ at every position.
- It is one draw of new tokens.
- "Mask alone" still runs on the rest of the window with some token at p + 1. So it does not show that B's identity is in the mask rather than passed by features the new token shares with B, nor whether the mask raises B or lowers its competitors.

## Wrong

**W1. Lines 112 and 116: both accounts "remain" and the reads "leave open why". The notebook has the same.**
Report: "That would give low gained with a real fall in B, the pattern read here. This alternative and 'the mask holds the answer' both remain, and each is an inference." And: "its lean to B partly survives with B gone, and the reads leave open why."

The pattern read here has a third feature, and a blind mask does not produce it: B sits 2.61 above its level under a mask not read for it, with B out of the text. A mask that only passes its token has nothing to pass there. Line 106 sees this, but line 112 then presents the two accounts as rivals.

Corrected wording: "A mask blind to the new token and passing B would give low gained and a real fall in B, but not B staying 2.6 log-odds above its own level under a mask not read for it, with B gone. A mask that holds the answer would give no part that needs both, and that part is 1.30. So neither account alone fits. On the mean, at W₀, at the second A, for this draw, about 0.62 of the 4.08 needs the mask and not the token in the text, 0.32 needs both, and 0.06 needs the text alone. This is arithmetic after the run, assuming the two tokens are exchangeable. It does not name what carries either part."

**W2. Line 110: "Under the held mask the forward does not follow the text."**
Line 66 of the same report has B falling 1.53 when it leaves the text, 9.5 sds over the bar.

Corrected wording: "Under the held mask the forward does not follow a new token into the text; it does lose part of B when B leaves."

## Overstated

**O1. Line 47: "The apparatus can show a copy of the new token, since both references read `mask_opens_route`."**
Both references have no token-read mask, so they show that the overwrite and the rule work. They do not show that a held, token-read mask could ever read a copy of a token it was not read for. Line 31 scopes this correctly.

Corrected wording: "The overwrite and the rule read a copy when no mask selects. Nothing shows a mask read on another token could read one, so low gained is not evidence against a route."

**O2. Line 152: the converse read described as the other half of the test.**
Report: "It holds the text fixed and changes which B the mask was read for, where this run held the mask and changed the text."

That is literally true, but both runs read the same off-diagonal cells: a mask read for one token over a text holding another. By exchange, the converse read should give about −8.3 for the mask's token (−6.794 − 1.53) and about −11.2 for the text's token (−11.447 + 0.23). So it fills the two unread cells above and checks exchangeability. It cannot separate the accounts further. Line 183's mask changed at one position only is the read that says where the surviving part lives.

Corrected wording: "The converse read fills the table's two unread cells and tests whether the tokens are exchangeable; expected about −8.3 and −11.2. It is not a new separation. A mask changed at one position, at the overwritten token or at the second A, is the read that locates the parts."

**O3. Lines 66, 108 and 112: the fall in B implicitly read as what a route would give.**
Line 112's "passing B" reads the 0.35 as route-shaped. Amendment 5's swap row, on the same windows and under the same held mask, reads −7.575 with the pair broken and B still in the text. So only 0.275 (sd 0.033) of the 1.533 needs the pair, which is 0.18 of it and 0.06 of the lean. The other 1.258 needs B in the text and not the pair match. The references run the other way: 0.82 and 0.83 of their full fall needs the pair. So what B's presence adds under this mask is not shaped like the target model's copy.

Add to line 66: "Only 0.28 of the 1.53 needs the pair, from Amendment 5's swap row. In the references about 0.82 does. So most of this fall needs B present, not A matched, and it is not a copy through the pair."

**O4. Line 74: "B's top-1 rate falls much further than its lean does", "0.65 of the lean survives in mean log-odds and 0.18 of the top-1 rate does."**
This sets a share of mean log-odds against a ratio of a threshold count, and the two are not on one scale. In the 39 windows, B falls from −1.20 to −5.15, a loss of 0.43 of those windows' own lean, against 0.30 in the other 161. So B does fall more where it was answering, but not by the 0.65 against 0.18 the sentence suggests. Reading the top-1 rate needs only a fall past the best competitor.

Corrected wording: "B is top in 7 of the 39. In those 39 it falls 3.95 log-odds, 0.43 of their lean, enough to lose first place in most of them. The two summaries are not on one scale."

**O5. Lines 5, 116 and 118: "so this one position does not yet say…", "This run does not show that here", "now has this read beside it".**
These understate rather than overstate, but they mislead the same way. A rule that fails to fire is not the reason, and the numbers do say something. Corrected wording for line 5's second half: "Neither of the user's two readings fires as filed. But most of the masked lean here does not need B in the text, so at this position a masked lean to a token is not by itself a read that the forward copied it. Whether that carries to the marker or to fine-tuned states is unread."

**O6. Line 135: "the cause was not tested", against line 66's "by arithmetic, because…".** See the controls section below. The two lines describe different things and should say so.

**O7. Line 122: "the result is not near them".**
The sds are over windows and keys for one draw of new tokens; the filing has no draw sd for these shares. Add "for this draw". This is unlikely to change anything: lost would need to move 0.88 log-odds.

**O8. Line 112: "Γ reads the target model's pass, so its following the token is expected on either account."**
This is true but incomplete, and it bears on what a mask holding B would mean. Γ here is a bidirectional transformer over the whole row (`src/commute/vpd.py`, `reports/reviews/2026-10-27_review_gamma-bidirectional-windows.md`). It can carry B to any position from its own inputs, with no copying in W₀ needed. Add: "and Γ sees the row in both directions, so a mask that favours B at the second A is not by itself evidence of a copying mechanism W₀ runs."

## Fine

- **Numbers.** Every number in the report's text and tables that `amend6.json` holds matches, as do the differences I re-derived: 8.60, 8.72, 8.28, 8.43, 1.53, 0.23, 2.81, 0.65, 2.55, 0.06, 0.51, 0.15, 4.58, 4.6 times, 9.5, 10, 7 and about 5 sds (5.5). The rows gave 167, 3, 6, 175, 39, 7, 7 among the 39, 0, 48. The sds reproduce. Commits, the sha256, job timing, file sizes and the notebook's "to 0.006" hold. The notebook's numbers are the report's.
- **The verdict by hand.** Gained 0.0532 is not over 0.6, so no route. Lost 0.3532 is not under 0.15, so no hold. That leaves `mixed`, and fp32 does the same. Both references read route. Rescaling the shares by the references' full range does not move it (next section).
- **Run against report.** The cells, the shares, the paired sds, the held mask and the own mask are what the report says. Γ and the forwards see the first 41 tokens only, so the appended target does not reach Γ (`route_cells`, `masked_lo_windows`, `masked_top1`, `gates_fp32`). B is nowhere in the window after the overwrite, and position 20 never held B. The fp32 row's lean is the bf16 lean, as filed; line 56 could say so.
- **"About two thirds survives" (line 9).** It holds up, although its floor, −11.64, has B in the text. Other floors with B absent give 0.63 (own mask) and 0.62 (new token absent). A floor scaled the way the references overshoot gives about 0.71. By quartile of `on` the survival runs 0.58 to 0.76. Lines 9 and 106 can stand, and line 106's inference can cite the same-text 2.61. Line 184 is literally true, but those two reads are partial stand-ins for the missing low end.
- **"Mask holds" and "mask is stale".** The report asserts neither, and rightly. Line 100 is right that table rows 1 and 2 cannot be told apart and that the own mask cannot isolate p + 1. The plan's sentence that the own-mask reads "say whether the held mask was stale at p + 1" cannot be met by this run. What can be said: on the same text, the change of mask alone moves the new token up 4.35 (sd 0.25) and B down 2.61, against 0.23 and 1.53 for the change of text under the held mask. The report never states this comparison. It is the clearest thing the own-mask reads show, and I suggest adding it at line 90.
- **Line 102.** Reproduced; "not tested" is fair.
- **Lines 37 and 76, "leans to B".** This matches `decisions.md`'s last row.
- **Line 116's row 112.** Correct.

## The two questions the brief asked

**Is a held mask read on one token a fair test of a replacement token?** Not as a test of whether a route exists. Low gained is expected whether or not a token-gated route exists, and no positive control shows otherwise (O1). It is a fair test of a narrower thing: whether this mask, read for B, carries a token it was not read for. It does not: 0.23 log-odds. Lines 68 and 112 already hedge this. Line 110's first sentences read as evidence against the branch, and should be scoped the same way.

**Does the run separate a mask that holds the answer from one that passes only the token it was read for?** The filed rule does not, and the report says it does not. The cells do at the two ends, as above. Neither pure account fits, and each describes a measured part: 0.62 and 0.32. What the run cannot say is what those parts are, and which of the user's branch actions, if either, they justify; that stays with the user.

## Are the controls' overshoots a problem?

No. They do not touch the verdict, the stops or the reading.

- **Where they come from.** Almost all of it is the B-absent level at the second A. Had `b_over` read the filed −12.2, lost would read 1.118, the top of the filed range. It reads −12.96, and `new_intact`, a separate row with other tokens, reads −12.90; the target model reads −12.99 and −12.93. So −12.9 to −13 is the level at this position, and the filed −12.2 was a read at the marker. Gained is the same 0.10, plus 0.03 from `new_over` reading −4.18 against about −4.4. Why the second A sits lower than the marker was not tested.
- **Which is arithmetic and which is untested.** Line 66's "over 1" is arithmetic: `b_over` is under the pair-broken level by 1.59 and 1.45. Line 135's "cause was not tested" should be narrower. Corrected wording: "The overshoot beyond the filed ranges is the B-absent level at the second A, −12.96 against the filed −12.2 at the marker, in two separate rows. Why it sits lower at this position was not tested."
- **What the controls are for.** They test that a forward following the text reads as following it, and no stop reads the ranges. Each passes by a wide margin.
- **Effect on the shares.** The lean is 0.82 of the references' full fall, so a share of 1.22 means "all of it". On that scale the mask's lost would be 0.29 and gained 0.04, and the verdict is still `mixed`, with lost still far over 0.15. A reader should not take 0.353 as a third of a full following; it is nearer 0.29 of one, if the masked forward's scale matches the references'. That is unread.
- **Line 143's table** also omits `new_intact`, −12.899, past the same filed range. Worth adding, since it is the second row that shows this is a level and not a B effect.

## Not checked

The figures (not viewed), `plot_amend6.py`, the Fable advice's text and any read needing the model. The two unread cells and any other draw of new tokens are also unread by me.
