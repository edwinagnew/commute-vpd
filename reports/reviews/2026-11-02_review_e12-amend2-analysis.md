# Review: E12 Amendment 2 analysis (`reports/analyses/2026-11-02_analysis_e12-amend2-rand32b-fall.md`)

*2026-11-02, independent check by an agent that wrote neither the report nor the code. No model load and no GPU run. Recomputation script and output: `~/commute-data/scratch/commute-lead/e12a2-check/recheck.py`, `recheck.out`. Line numbers are the report's unless another file is named.*

**Verdict.** The numbers are right and the code measures what the report says, with one exception on precision (M2). The filed verdicts stand. The problems are in the reading. The opening paragraph and "What we learned" state a two-table conclusion that only the second table supports (H1). They also present the outside block as a finding about thousands of subcomponents, when it returns about what its share of the change would predict (H2). The notebook's section, cell 34, carries the same wording, and three figure titles carry parts of it.

## High

### H1. "Keeps its behaviour but not its subcomponents" is false on the first table, and no state shows the term kept with the set lost

Lines 5 and 198; notebook cell 34, "What we learned".

Line 5 says that "on these two tables and this position, a student keeps its fine-tune's behaviour but not its subcomponents or weights". Line 198 says "a student's nats do not imply its subcomponents". Three things read against that.

- **The first table.** Students keep the term (2.0% and 2.4% lost) and most of the set: J = 0.905 and 0.833, against 0.550 between that table's two fine-tunes. P2's test was filed for the second table only, but run on the first it would read "same". Line 198's own second sentence says this, so the paragraph disagrees with its first sentence.
- **The direction of the evidence.** The state with a kept term has the higher overlap, and the state with a lost term has the lower. No state here keeps the term and loses the set, so "nats do not imply subcomponents" has no counter-example if "nats" means the term. It holds only if "nats" means the target's log-odds or P(target).
- **"Means nats only".** On the second table the student does not keep the term in nats: it loses 16% to 20%. Amendment 1's report, line 126, says it holds there "only in units of the gain".

"Behaviour" is also looser on the first table. P(target) is 0.629 against the fine-tune's 0.585, and 0.571 against 0.544 (`.students[].a.p_cells.T1`, `new_vs_existing_amend1.json` `.c0[].nothing.p_cells.T1`). Line 198's "within about 0.003" holds on the second table only, where the differences are 0.0023 to 0.0031.

**Corrected wording, line 5, last sentence.** "For `GOALS.md`: on the second table a student keeps its fine-tune's P(target), but loses about a fifth of the re-switching term and part of the switched-on set. On the first table it keeps the term and most of the set. On both, its weight change is far from its teacher's. So P(target) does not imply the term or the set, and the cause stays open."

**Corrected wording, line 198, first sentence.** "On these two map tables, this card and this position, a student's P(target) does not imply its re-switching term, its switched-on set or its weights. The term and the set fall together here, on one table of two."

### H2. "Nearly all of it comes back from thousands of subcomponents" has no reference, and its number of carriers was not measured

Line 5; notebook cell 34, "The answer"; the block figure's title, `plot_amend2.py` `blocks()`.

- **The reference that is missing.** The outside block holds 97% to 98% of all absolute change on the second table (`amend2_summary.json` `.states[].outside_share_of_all_change`) and returns 87% to 88% of the fall. The set members hold 2.4% to 3.0% of the change and return 3.0% to 4.2% of the fall. So nats given back roughly follow where the change is, and the set members return slightly more per unit of change. Lines 82 to 84 rightly say the label is a remainder, but they never set the 87% beside the 97%. The reader takes "outside" as a localisation when it is close to the null.
- **"Thousands".** That is the block's size, 9,853, not a count of carriers. The largest group written back is 300, which returned 31% to 34%. Line 101 says the ladder cannot separate spread from joint action, so "thousands" is not available as a finding.
- **The notebook goes further.** "Thousands of other subcomponents, whose values move sharply in single windows" joins nats given back to large moves, and nothing tested that link. Only about 2,500 of the 9,853 have any entry over 0.3.

**Corrected wording, line 5, second sentence.** "The fall survives a second batch order. About seven-eighths of it comes back from the block of everything outside the switched-on sets, which also holds 97% of the change in values, so this says little about which subcomponents carry it. The 300 most changed give back a third."

## Medium

### M1. The control says the set does not hold the term either, and the report misstates E12

Lines 82, 198, 211, 217.

The control's top 30 contains 19 of 27 and 17 of 21 of the fine-tune's set, and its smallest change is 0.35 (`.ladder_control[].top` against `.fine_tunes[].S_on`). Writing it back from W₀ at the trigger gives 1.2% and 1.5% of the term. So under this read, the set members alone hold almost none of the term itself, not just of the fall. "The 0.3 rule does not read what the fall is made of" is therefore not specific to the fall, and the report does not make that connection.

Line 217 says the term was one "that E12 read through sets of 21 to 52 members". E12 did not read this term through sets. Its report says of the mapping that "the behaviour runs through subcomponents it keeps off" (`reports/analyses/2026-10-20_analysis_new-vs-existing-signatures.md`, lines 12-13 and 164-165). So the kept lean, and the filing's "the term lives in sets", ran against E12's own reading. The miss stands as filed, but line 211's "cannot explain" should say this.

**Corrected wording, line 217.** "The top 30 of 9,881 changes hold 17 to 19 of the fine-tune's set members and give 1% to 2% of the term. E12 already read mapping behaviour as running through subcomponents the rule leaves off, so a flat ladder is less surprising than the filing's guess made it. Spread and joint action cannot be told apart here."

**Addition to line 198.** "The same holds for the term itself on the ladder's control, so this is not a property of the fall alone. It rests on one group, written back alone."

### M2. Both reads in the hook and weights pair run under bf16, so the pair does not test precision

Line 48; heading at section 1; notebook cell 34, "rather than from rounded weights"; `predictions_amend2.json` `.measurements.L01_hook`.

`hook_l01` calls `route_cells`, then `compute_ci`, which runs under the same bf16 autocast (`src/commute/vpd.py:179`). Training's forward is also under autocast (`src/commute/finetune.py:60`). The offset's matrix products run inside that context too. The report's phrase is right that the hook form is "the forward it trained on", but its contrast implies only the weights form is under bf16.

What the pair varies is how a small dW is rounded: inside W₀ + dW, or separately beside W₀. That is a useful check, and it passes at 0.0069 nats or less. But no read of Γ here runs in fp32 on either side, and `masked_logits` is also bf16 (`local_faith.py:55`). The "Not run" entry covers fine-tunes only.

**Corrected wording, line 48, last two sentences.** "So rounding dW into the weights does not carry the fall. Both forms read Γ under bf16 autocast, so this is not a test of precision as such. No read here is in fp32, on either the student or the fine-tune, and a fine-tune has only the weights form."

**Add to "Not run":** "Γ in fp32 on any state." Given the standing rule on precision pairs (2026-11-02), the lead should decide whether this is worth one cheap job.

### M3. "On the most changed matrix only" is not read

Lines 140, 155, 163; `plot_amend2.py:263`; notebook cell 34, section 7.

The max-matrix ratio takes each model's own maximum over matrices, so student and teacher may not be on the same matrix, as line 153 says. Per-matrix sizes of students were not read. So the data show that the ratio is below 1 on one measure. They do not show that a student is smaller on one matrix and larger on the rest. The teachers' top three matrices are close in absolute size: 0.442, 0.438, 0.420 on seed 0, and 0.448, 0.446, 0.429 on seed 1 (`recheck.out`). A student could be below its teacher on several.

Line 155's "it must be larger than its teacher elsewhere" is sound only as "on at least one matrix", and it follows from the totals, not from the 0.0226 cap.

**Corrected wording, heading and figure title.** "On the largest-matrix ratio, yes; on total size, no."

**Corrected wording, line 155.** "The ratio of each model's own most changed matrix is under 1. Since the student is larger in total, it must be larger than its teacher on at least one matrix; how many was not read."

### M4. The set comparison has one wrong statement and a sensitivity it does not report

Line 65; heading at section 2; `plot_amend2.py:146`.

- **Wrong.** "Second-table students overlap their own fine-tunes at 0.57 and 0.68, between the other table's 0.26 and the other seed's 0.66." 0.682 is above 0.655, which is why the rule needs "partly" rather than a bracket. Corrected: "at 0.57 and 0.68, either side of the other seed's 0.66 and well above the other table's 0.26".
- **Sensitivity.** "Lower" is the filed verdict, on order-0 students, and stands. But the report's own order-1 student reads 0.773. With it, the gap is 0.833 − 0.773 = 0.060, under the noise of 0.111, which would read "no different". The heading's "less than on the first", and line 198's "only 0.57 and 0.68", leave out 0.61 and 0.77. Add: "Lower by the filed order-0 rule; a second-order student reaches 0.77, and with it the gap falls under the noise."
- **Not accurate.** "Since the first table had no second batch order". Amendment 1 has `rand32_s0_o1`, with L01 −8.2284 and saved offsets (`results/large/amend1/rand32_s0_o1.pt`). Line 218's "about ten times the first's" relies on it. What is missing is a set read on it, and "Not run" should say that.

### M5. "Single-window moves" is not what the data show

Lines 119 and 277; `plot_amend2.py:214` ("large only in single windows"); notebook cell 34, "The answer" and section 5.

On the second table, a subcomponent with an entry over 0.3 has 3.2 to 3.3 such entries on average, median 2, up to 21 of 97. Only 31% to 33% have exactly one (`recheck.out`). The counts and the 59% are right. "Single-window" is not.

Two further readings from the saved files are cheap and belong in the report:

- **Windows.** The top 5 of 97 hold 16% to 18% of these entries and 14% to 15% of the absolute change, against 5% if uniform. That is uneven, but not a few windows.
- **Sign.** The fine-tune is above the student on 62% to 66% of these entries on the second table, and 41% to 44% on the first. This matches the net sign in section 6, and it is unfiled.

**Corrected wording.** "59% of the absolute change sits in per-window moves larger than 0.3, typically in two or three of the 97 windows for a given subcomponent." Whether those moves return the nats was not tested, and the sentence should say so.

The figure's 0.3 line is labelled as the rise over W₀, on an axis of fine-tune minus student. These are different quantities. The row "subcomponents with mean change over 0.3" is close to true by construction, since every outside subcomponent has a rise of at most 0.3 on both sides.

### M6. "Not close" has no reachable reference, and the filed caveat is dropped

Lines 5, 188, 192; `predictions_amend2.json` `.predictions.P5.blind_to`.

The arithmetic is right. But the filing says that a low cosine "can be what a student cannot express with Delta fixed, so this bears on 'the student reaches its teacher' only loosely". The report omits this. Teachers also trained for 10 or 30 steps against the students' 1,000, and Amendment 1's student with nothing to learn still grows. So the off-teacher length may be drift. Line 275 mentions drift, but only for size.

"Not its weights" and "removes its one direct test" need this beside them.

**Addition to line 188.** "How high a cosine a student could reach with Δ fixed was not read, and part of the off-teacher length may be drift over 1,000 steps. So this shows the weights differ, not that the student failed to reach anything it could reach."

## Low

- **L1. Stale corrections table.** Lines 159-169 and 231, and the notebook's "What next". Commit 4aba4934 applied them at 17:19, three minutes after this report's commit. The "now reads" column no longer matches lines 99, 124 and 188 of Amendment 1's report, notebook cell 31 or `plot_amend1.py:169,174`, and "the one open job" is done. Replace both with one line naming the commit.
- **L2. "Already on at W₀".** Line 121. The group is defined by the fine-tune's mean over 0.3, and W₀'s trigger values are not saved, as line 205 says. Say "outside subcomponents with a fine-tune mean over 0.3". "Not the place" also speaks to change, not effect. The 5.9% to 6.7% itself reproduces.
- **L3. Kept and dropped ranges.** Line 67. "Under 0.03" is the block mean. Single kept members move by up to 0.094. Dropped means fall by 0.027 to 0.132 at order 0, but from 0.022 at order 1 on the same table and from 0.004 on the first.
- **L4. The dotted line.** Line 103. "About as much" holds only near k = 300, at 1.1 to 1.2 times. The give-back share is about 1.5 times the change share at k = 100, 1.1 to 1.4 times at 30, and 5 to 11 times at 1, so the most changed return more per unit. The ladder figure's "the control reads the same" is true of the label only: the control sits below the student at every k.
- **L5. Unsupported sentence.** Line 204, "small groups recover far less than their share would suggest". The control's share of change was not computed and cannot be, since W₀'s values are not saved. Cut it.
- **L6. Filed clause versus evidence.** Lines 136 and 225. The clause fires, and quoting it is right. But "spread" fails the ladder's control and "lowering" fails the first table, on the report's own reading. Add: "recorded as filed; this run supports neither word as a cause."
- **L7. Omitted blind spot.** Line 205 leaves out P3's filed "per-window sign, since nats are means". The fall per window is not saved. `local_faith.masked_lo_windows` exists, so add it to "Not run".
- **L8. Off-teacher "similar".** Line 194. 1.85 to 1.87 against 1.62 to 1.65 is 14% apart. The length along the teacher, in absolute terms, is 0.80 to 0.84 against 0.52 to 0.55, 1.5 times. Report both, since the sentence's "may largely follow from a larger teacher" rests on that gap.
- **L9. Small numbers.** Line 204's 1.120 is 1.119, and the gap to the whole includes dead subcomponents, which are in "every value" only. Line 157's 1.75 is 1.76. "Under 2%" is 2.04% on second-table seed 1 at order 1. Line 23's "writing back every value" ran on the first student only, as filed.

## Checked and found sound

- **Provenance.** Both sha256 values match. The commit, the card and the 12 min 39 s run time match the log. `amend1.py`, `amend2.py`, `recruit.py` and `local_faith.py` are unchanged since 8d69671. Amendment 1's file predates the run, so `amend1.OUT` protected it.
- **Tables.** Every number in the tables of sections 1 to 8, the term arithmetic, the shares and the along and off-teacher columns reproduces from the JSON, the summary and the log. Set table: every J and block count is consistent with the set sizes, and the kept means reproduce from the trigger files. The matching 0.554, 0.542 on two rows is real, not a copying error. Teacher sizes agree with the file to 15 digits. All 22 checks pass, with differences of exactly 0.
- **Rules.** P1 to P5 and `against_reference`, re-applied by hand, match `.score`. The code's rules match the filed text. The leans and the quoted clauses are verbatim.
- **Code.** The restore writes only at the trigger: `hybrid` over `with_trigger` leaves other positions as the student's. The four blocks partition the alive set. The L01 read uses released components, Δ out, and depends on the state through gates only. The hook rebuild's offsets have the right layout and are removed before `masked`. Dropped, kept and added match the two `S_on` lists. The drop-list claims hold: order 1 is a subset of order 0, and four members drop on all four.
- **Saved values.** They are fp16 and off the bf16 grid (about 11% to 13% on it, near chance). The "under 0.001" rounding claim holds. `trigger_summary` reproduces on the six states for the counts checked.
- **Behaviour on the second table.** P(target) is within 0.0031. Held-out `kl_other` against W₀ is 0.093 and 0.100 against the teachers' 0.097 and 0.104. A held-out KL to the teacher was not read in this run.
- **Conclusions that hold.** The fall is not batch order. The fall survives the hook form. The set members return little, in both orders of writing back (the gap between at-trigger and outside is 0.079 and 0.060). The total-size correction to Amendment 1 is right. The section 8 caution, added after the filing and labelled so, is fair.
- **`check.py`.** 0 errors on the report; its 52 flags are style.

**Not checked.** I ran no L01, block or ladder forward again: those are taken from the JSON and log, which agree. I looked at five of the eight figures (ladder, blocks, size, outside, alignment); the other three I read in code only. The E42 and copy-table total sizes "not on file" were taken from the report and Amendment 1's file keys.

## What the lead did

Applied 2026-11-02 to the report (R), notebook cell 34 (N, rewritten whole with `nb-edit set`), `plot_amend2.py` titles (F, figures redrawn from the saved summary) and the TRIED row (T). No number or filed verdict changed; `check.py` on R: 0 errors.

- **H1.** Applied: R line 5 last sentence and "What we learned" use the corrected wording, plus the first-table P(target) figures and "no state keeps the term and loses the set"; N "What we learned"; T headline.
- **H2.** Applied: R line 5 second sentence (corrected wording), a new reference paragraph in section 3 and its heading; N "The answer" and section 3; F block title; T headline.
- **M1.** Applied: R section 4 (control's top 30 hold 17 to 19 set members, 1% to 2% of the term), "What we learned" (addition), the control-ladder bullet in "Surprises" (corrected line-217 wording); N section 4 and "Surprises".
- **M2.** Applied: R section 1 last sentences (corrected wording) and heading, F fall title, N section 1. "Not run" says "Γ in fp32 on any state: being run as Amendment 3" with no result claim (lead's decision); "What next" names Amendment 3.
- **M3.** Applied: R section 7 heading and "Reading it" (corrected wording, plus the teachers' top-three sizes), N section 7, F size title.
- **M4.** Applied: R section 2 heading and reading (0.57 and 0.68 "either side of" 0.66, order-1 sensitivity 0.060 under 0.111, `rand32_s0_o1` exists with no set read), scorecard line and N section 2; "What we learned". "Not run" adds the set read on `rand32_s0_o1`. F sets title shortened to "Partly; lower on table 2" because the longer title was clipped.
- **M5.** Applied: R section 5 (per-window moves, typically two or three of 97 windows, not tested whether they return the nats, top-5 window share, sign shares, the 0.3 line's different quantity), the "Not run" bullet; N section 5; F outside title.
- **M6.** Applied: R section 8 (caveat added after the 83% to 91% sentence, with the filing's own words); N section 8.
- **L1.** Applied: R corrections table replaced by one line naming commit 4aba4934; the "one open job" sentence replaced; N "What next".
- **L2.** Applied: R section 5 ("outside subcomponents with a fine-tune mean over 0.3", "speaks to change, not effect").
- **L3.** Applied: R section 2 (block mean under 0.03; single kept members up to 0.094; dropped-mean ranges at order 1 and on the first table).
- **L4.** Applied: R section 4 (dotted line ratios, control below student at every k), N section 4. F ladder title "the control is flat too" in place of "reads the same".
- **L5.** Applied: R "How compelling": the unsupported sentence is cut.
- **L6.** Applied: R section 6 and "What next" ("recorded as filed; this run supports neither word as a cause"); N section 6 and "What next".
- **L7.** Applied: R "Not run" (the fall per window, `local_faith.masked_lo_windows`); N "Not run".
- **L8.** Applied: R section 8 caution (14% apart; 0.80 to 0.84 against 0.52 to 0.55, 1.5 times); N section 8.
- **L9.** Applied: R 1.119, 1.76, "2.04% at most" (also "about 2% or less" in N and R), the dead-subcomponent note, and "on the first student only, as filed". Nothing skipped.

Not changed, by instruction: `predictions_amend2.json`, `amend2.py`, any number, any verdict. The research log is untouched; `docs/research_log/PENDING.md` carries a line for the E12 card.
