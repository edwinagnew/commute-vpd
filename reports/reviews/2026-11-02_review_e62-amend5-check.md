# Review: independent check of E62 Amendment 5's head sweep

*Independent checker; did not write or run this work. No GPU job, no model loaded. Under check: `reports/analyses/2026-11-02_analysis_e62-amend5-head-sweep.md` ("the report", line numbers are its own), `notebooks/E62_induction_cue.ipynb` cell 45, both figures. Recomputed from `results/large/amend5_windows.pt`, `amend5.json` and `predictions_amend5.json`, with my own key bootstrap (2000 draws, seed 1): `~/commute-data/scratch/regating-lead/e62a5-check/recompute.py`, output `recompute.out` beside it.*

## Verdict

**The filed verdicts and the sweep's numbers hold. The verdict `thinned` follows from the filed rule, and `through_gates` follows from its rule. The main error is in the lead sentence's "a quarter of the effect".** That compares falls read with the mask frozen against a lean that is almost all change of mask. On the only same-kind scale the saved rows give, the falls overshoot rather than fall short.

Five other errors need correcting (F2 to F6) and four small ones (F7 to F10). Separately, I would state the conclusion "the mask does not show the mechanism, so no head-level anchor" less strongly, as the sweep freezes the mask and so cannot see the channel the report says the pair's effect travels through (R1). No filed result moves.

## What held

Recomputed from per-window rows; the file and the JSON differ by 0 on T, M, Tb and Mb for all 24 heads.

| claim | report | recomputed |
|---|---|---|
| every T, M, Tb, Mb, pair part in the table (l.71-94) | as printed | agrees to the printed digits |
| filed verdict by hand: no other head has M ≥ 1; seven fractions within 0.24 of 0.190 | `thinned` | `thinned`; widest are (3,5) −0.024 and (2,3) 0.364, band −0.050 to 0.430 |
| floors 0.25, 0.5, 1.0 | `mixed`, `thinned`, `thinned` | same, from `copying.sweep_reading` |
| control: (2,4) with every mask at 1 | 4.91, 1.02, r 0.99, lean 7.01 | 4.907, 1.019, 0.987, 7.010 |
| re-read and do-nothing | 0, 0 | `max_diff` 0, `exact` true, both repeats 0 |
| tie: T(1,1) − T(2,4) | +0.15, key sd 0.15, range −0.12 to +0.44, 60% of windows | +0.154, 0.150, −0.14 to +0.45 at my seed, 59.5% |
| ratios (1,1) over (2,4) | 1.03 target, 0.20 masked | 1.032, 0.197 |
| E62's scores | prefix 0.42 and 0.0002; previous-token 0.67; top-1 0.0095 and 0.013 | 0.424, 0.0002, 0.668 (`run.json`); 0.0095 and 0.013 (E62 analysis l.74) |
| summed falls, arithmetic | 1.06, 4.34, 0.24; positive 1.49; pair parts 1.37 and 13.58 | 1.058, 4.341, 0.244; 1.486; 1.368 and 13.582 |
| swap | −7.30, −7.58, −11.66, −11.64; token shares 0.063, −0.004; mask parts 4.36, 4.07 | same; `token_reading` by rule `through_gates` |
| depth table (l.116-121), negative heads (l.182), (3,5) pair part 0.67, −7.30000003 closed | as printed | same |
| predictions committed unchanged before the run | sha `f1b4dace...` | file sha matches; last edit 327d884 at 16:57:33 (review finding 1, recorded in the plan), job logs from 16:58:51 |

The reference to Amendment 2 is right: it refuted strict series on net shares at the marker (its l.130). The 0.81 correlation of the two target falls is also more telling than the report says: six other heads I read correlate with either at 0.00 to 0.37. It still does not separate series from a shared window-level strength. "59 s" is unchecked, as `sacct` was down.

## Errors, ranked

### F1. "A quarter of the effect" compares two kinds of read (l.5, l.110, l.112, l.199, notebook "How much of the masked copy")

The arithmetic is right. The comparison is the one the plan's Blind-to rules out ("a fall of M against a lean of 4.34 is not a share of the copy"), which the Amendment 4 check's F4 corrected, and which this report's own item 4 (l.18) faults in the 1.40. A fall holds the mask. The lean is 94% to 100% change of mask (4.07 to 4.36 of 4.34), so no frozen-mask fall could account for that part, whatever the heads do. "Three quarters unaccounted for" is therefore mostly true by construction.

The same-kind denominator is the token part with the mask held, at the mask the falls were read under:

| | value | with key range |
|---|---|---|
| token part under the intact mask | 0.275 | |
| 24 net falls over it | **3.84** | 3.1 to 5.0 |
| (2,4)'s fall alone over it | **3.33** | 2.6 to 4.4 |

So with the mask held the falls do not fall short of the pair's effect: one head moves over three times what breaking the pair moves.

**Correction.** Replace the plain-words sentence's "together the falls come to a quarter of the effect, which sits almost wholly in the mask" with: "the pair's effect sits almost wholly in the mask, which a head switched off with the mask held cannot reach." Replace the headline at l.110 with "**The falls cannot account for it: they hold the mask and the lean is in the mask.**" Keep 0.24 only as a printed flag that decides nothing, and give 3.84 beside it. Delete l.112's "accounts for a quarter" and the clause in l.199. "Most of the masked copy is not there to be removed by switching off one head" may stay, with the swap as its reason and not the sum.

### F2. The reference is mismatched, and its stated cause was not measured (l.110, notebook)

The sentence sets masked **falls** (0.24) against target **pair parts** (1.99). `amend5_reads.json` already holds the like figure, unreported:

| sum over own lean | target | masked |
|---|---|---|
| net falls | **0.94** (key range 0.61 to 1.25) | 0.24 |
| positive falls | 2.37 | 0.34 |
| pair parts | 1.99 | 0.32 |

"The reference overshoots" holds for two rows out of three. On net falls the target does not overshoot. "Because falls overlap when heads act in series" is a cause: both heads off was not read at this position, l.104 calls the chain a presumption, and Amendment 2 refuted strict series at the marker. The notebook's "since heads in series overlap" has no hedge at all. Also, "at most about a third" (1.49 of 4.34) leaves out the positive pair parts, 1.65, or 0.38.

**Correction.** Give all three rows, and write "falls need not add; heads in series would be one reason, unread here".

### F3. "The pair parts agree with the two groups" is wrong (l.67)

Group one: (2,4), (3,4), (2,3). Group two: (1,1), (0,4), (1,2), (0,2), (3,5).

| | group one | group two |
|---|---|---|
| target pair part | 4.75, **0.08, 0.07** | **5.09**, 1.73, 1.55, 0.43, 0.67 |
| masked pair part | 0.91, 0.20, 0.21 | **0.19**, −0.03, 0.00, −0.01, 0.00 |

The target's pair parts do not divide along the groups, as the two largest fall one each side, and rank order across all 24 is 0.35 between forwards. The masked ones nearly do, except (1,1). "Both heads" in the next sentence has no antecedent in that paragraph.

This matters more than the wording. (3,4) and (2,3) are in the upper group on target falls that are almost all generic: T 0.73 with Tb 0.65, T 0.52 with Tb 0.46. So their fractions of 0.23 and 0.36 are not fractions of a copying fall.

**Correction.** Delete the sentence. Add: "Two of the upper group's three have almost no pair part in the target model, so their fractions do not read copying. Of the six heads whose target fall has a pair part of 0.4 or more, only (2,4) keeps over 0.04." That last sentence is mine and post hoc.

### F4. "Account D is not live" contradicts the report's own blind list (l.129, l.195, notebook)

Fable's advice words D as "the copy is token-carried through whatever is open". The report narrows it to "the lean comes through the tokens with either mask". Then l.135 says "the intact mask on pair-broken tokens may still draw B from the tokens through whatever the mask opens", which is D as first worded. The filed token reading is account D's read, so `through_gates` is correctly reported. It reads only dependence on the pair, as B stays in the tokens on all four rows.

**Correction.** "The part of account D that needs the pair in the tokens is out. Whether B's identity is read from the tokens through what the mask opens is unread, and B overwritten under the intact mask is the test."

### F5. "Almost all" of the 1.40 being the change of mask understates it (l.18, notebook "most of it")

Recomputed: 3.425 = **4.066** (mask's part on pair-broken tokens) − **0.641**. The mask's part is more than all of it. Holding the intact mask on both points, head off (−8.216) sits **0.64 below** pair-broken tokens (−7.575), key range 0.52 to 0.75, where the target's head off sits 2.02 above. On the like-for-like read the excess has the opposite sign.

**Correction.** "All of it and more: with the intact mask on both points, head off sits 0.64 below pair-broken tokens, where the target model's sits 2.02 above." Rounding: the difference is 3.43 and the excess 1.41 (Amendment 4's report has 3.43). The filed text's 3.42 and 1.40 stay as quotes.

### F6. Precision is ruled out further than it was read (l.49, l.96, l.149, l.187)

`amend5.py` reads Γ in fp32 on **intact windows only**, and only (2,4)'s fall and the intact level (`g32 = R.gates32(w)`). Every other fp32 row, the 24 heads' M32 and Mb32 and `fp32_lean`, runs an fp32 forward under **Γ's bf16 masks**. The pair-broken mask was never read in fp32, so neither the swap, nor the lean's mask part, nor `a` has a mask-precision read. Meanwhile a mask entry moves 0.83 between passes, at an unknown place, and `gate_max_abs` covers all entries, alive or not.

So "Precision does not matter either" holds for (2,4)'s fall and the forward's precision. "Touches no reading here" (l.187) and "Both precisions give the same verdict" (l.96) are too broad, as `through_gates`, which carries the report's conclusion, rests on the unread mask. Given `feedback-run-down-anomalies-first`, the 0.83 should not be closed by a size on one fall.

**Correction.** State the scope at l.49 and l.149 ("out for the forward's precision, and for Γ's on (2,4)'s fall on intact windows"). At l.187 write "its effect on that one fall is 0.003; Γ on pair-broken windows in fp32 and the swap under it were not read". Add both to Not run.

### F7. Post-hoc list is short (l.172)

It names the two groups, the masked ratio and the depth table. Also noticed after the run with no filed rule: the summed falls set against the lean as a reading (the filed flag `sum_falls_over_lean` decides nothing), the general quiet on pair-broken windows (l.114, called "no value filed" only at l.176), the previous-token chain (l.104), and the inference that (2,4) "draws little on (1,1)" (l.106; the ratio is listed, the inference is not). The plain-words paragraph and The answer use the first and the quiet without a label.

**Correction.** Add the four, and mark the sentence in the plain-words paragraph.

### F8. Two accounts of the label (l.15 against l.159)

l.15: "The label on (2,4) comes from the prefix score, and a score is not a fall." l.159: "E62's causal label". Both are partial. H was picked by the prefix score (0.42 against a bar of 0.077), and passed a causal control, top-1 0.8247 to 0.0095 with it off. (1,1) reads 0.013 on that same control, and E62's analysis said so (l.74-80). So part of reading 1 was on file, and l.100's "nothing had measured it" holds for log-odds at this position only.

**Correction.** "H was picked by the prefix score and passed a causal control that (1,1) would pass as well; E62's analysis noted that on top-1, and this is the same near tie in log-odds at the second A." The second A does not settle the marker, where Amendments 2 and 3 read, and (2,4) is the larger there on the fine-tunes' gain (0.47 to 0.56 against 0.23 to 0.29).

### F9. "Agree to 0.002 on every head" (l.96)

The table's Mb column does not: the largest is 0.0043. M agrees to 0.0015. **Correction.** "0.002 on intact windows, 0.004 on pair-broken."

### F10. Provenance (l.222)

"Results committed at `27d425c`": that is a merge whose own stat is `interchange_67m`. `amend5.json` was added by **`c3687ae`** (16:59:32). **Correction.** Name `c3687ae`.

## Readings I would state differently

**R1. "Does not show" and "no head-level anchor" reach past a frozen-mask read (l.5, l.9, l.157, l.194).** The report's own finding is that the pair's effect travels through the mask, and every sweep row freezes the mask, so any effect of a head through Γ is excluded by design. Masks recomputed with a head off are the read that would test whether the lean depends on (2,4) or (1,1) through Γ. At the marker that read moved W₀'s fall from 0.04 to 0.39, with (3,5) at 0.21, so it is not empty and not specific. It sits in Not run, and What next does not propose it. The advice's words for this branch were "not shown either way at head level", and for a uniform attenuation that did not occur. "On this model" in the lead also generalises from one position, one seed and one marker's windows, though l.167 says so. I would write: "With the mask frozen, the masked forward's falls do not reproduce the target's at head level, at W₀, at the second A. Whether the mask depends on these heads through Γ is unread, so a head-level anchor is not found rather than absent." I would add that read to What next beside the two proposed.

**R2. "Costs little" undersells the one thing the mask keeps (l.9, l.114, l.176).** (2,4)'s 0.92 is 12.7 key sds, 4.8 times the next head, 3.3 times the token part, and about 100 times its own fall under the pair-broken mask (0.008). Its key range, 0.76 to 1.06, holds the carrier bar's M of 1.0, though the rule's other bars exclude it. "Barely reacts on any windows" is true of pair-broken windows and of 23 heads. A fairer sentence: "Under the intact mask the forward reacts to one head at a fifth of the target's fall, the head E62's prefix score picked, and little to the rest; under the pair-broken mask it reacts to none." The fall itself may be a removal or an injection, which the report notes. This does not rescue the mechanism reading, as (1,1) is lost, but it is the pattern to explain. It is the reverse of the filed `labelled_head_singled_out`, and it has no filed word.

**R3. The inference from (1,1) to (2,4) is conditional twice (l.106).** It assumes the chain, which l.104 calls a presumption, and that a mean replacement of (1,1) under a frozen mask removes what (1,1) writes. Block 1 is quiet in general (0.027), so low (1,1) may be the general quiet and not a statement about (2,4). The two masked falls still correlate at 0.56 over windows (0.49 over keys), which the report leaves out. I would write "fits", not "so".

**R4. l.201's proposed read already has a saved bound.** With the intact mask held, head off is 0.64 below pair-broken tokens. If effects add, at least about 0.64 of (2,4)'s 0.92 stays with the pair broken in the tokens, the opposite of what M − Mb's 0.91 suggests. This is not a measurement of the fall; it gives the proposed read an expected value, as the project requires.

**R5. Further blinds for `through_gates` (l.133-139), and one to tighten.**
- A head sweep reads only attention outputs. A mask that raises B through MLP, embedding or other non-head subcomponents, or that encodes B's identity directly, is invisible to it. l.146's "out for any single other head" should say heads only, and Not run should say so. B overwritten under the intact mask tests this too.
- l.136 says the mismatch at the overwritten position "has no size here". It has a joint bound: each swapped row lies within 0.28 and 0.02 of its matched row, so mismatch plus token effect is small unless the two cancel.

**R6. "Copying by this test" (l.17) needs its reference.** It is a fair reading of 4.75, and not a floor effect, as heads move B by over 2 on those same windows. But the test is not selective: (0,4) reads 1.73, (1,2) 1.55, (3,5) 0.67, and the sum is 13.58. I would write "depends on the pair, as do parts of several other heads' falls".

**R7. "The tie is real" (l.100, l.102).** The data show no separation: gap +0.15, key sd 0.15, (1,1) larger in 85% of key draws. "No separation, within about ±0.4" says what was measured.

## Figures

**`amend5_heads.png` shows what is said.** Band edges, the floor line and the points match the files; all seven compared heads lie in the band, and (2,2) lies left of the floor line and under it. (0,2), named as group two at l.57, is unlabelled in the zoom. Bars are window sds while the prose quotes key sds; the panel says so.

**`amend5_swap.png` is right, but leaves out the two references the prose leans on.** l.131 calls the every-mask-at-1 forward "the reference", and neither −4.37 and −11.38 nor (2,4) off under the intact mask (−8.22) is drawn. That last point beside −7.58 is F5's and F1's finding, and I would add both, under `/writeup`'s one figure per claim with its reference on the same axes. It carries no error bars; the key sds on the levels are 0.17 to 0.33, though the differences it is read by are paired.

## Notebook cell 45

It carries F1, F2 (unhedged), F4, F5 ("most of it") and R1, R2. Also "blocks 0 and 1 hardly register, at under 0.03" follows a sentence on pair-broken windows, where block 0 reads 0.043; 0.03 is the intact figure. Correct it with the report.

## Not checked

- Γ's and the forward's internals beyond `amend5.py`, `amend2.Reads`, `copying.ablated` and `copying.sweep_reading`; the code review covers them, and the control and exact re-read support them.
- `amend5_vectors.pt`; "59 s" and exit 0 beyond the log's stamps, as `sacct` was down; the sentence-level `check.py` flags (0 errors, 69 flagged, not read as findings).
- Nothing was recomputed that needs the model, so R1's and R4's reads remain unmeasured, and F3's last sentence, F1's 3.84 and R2 are this check's own post-hoc readings of saved rows.
