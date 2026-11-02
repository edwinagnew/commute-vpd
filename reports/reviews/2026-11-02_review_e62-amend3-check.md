# Independent check of E62 Amendment 3's analysis

*2026-11-02. Fresh agent; wrote neither the analysis nor the run. No GPU, no model forward, $0. "The report" is `reports/analyses/2026-11-02_analysis_e62-amend3-gates-head-off.md`. Recomputation: `~/commute-data/scratch/regating-lead/e62-a3-check/recompute.py`. Read vectors of block 2's `o_proj`, from the released checkpoint's weights only: `vread.py`, same folder. Output beside each.*

## Verdict

**The numbers, the code and the filed verdict are right; two conclusions in the report need changing.** Every number I recomputed matches. `amend3.py` measures what the plan and rules say, and the frozen and recomputed paths differ only in the masks. The filed rules give `head_more_than_target` with the lookup `within`, as reported.

**My own reading, reached before weighing the report's.** Under mean ablation, the masks that produce the copying map's gain change with head (2,4), and that change is needed for most of the fall. That change is absent on the lookup. Nothing in the read says what carries the gain inside the masked forward, and nothing reads how much of the head's output W₀'s mask passes.

**The report is careful on its main point:** "the masks follow the head" is kept apart from "the old mechanism". It should change these two things:

1. Line 13 says the "other way" reading is no longer live. The frozen share with the sampled slot open is the evidence for that account's carrying claim, and the report says so itself at line 125.
2. "W₀'s mask shuts this head" (lines 11, 39, 72, 184, 199) rests on eight subcomponents. They hold 5.4% of that slot's read weight among the 524 alive in that matrix. The rest was not read.

## Findings that change a conclusion

**F1. The "other way" account is not dead; the accounts are not exclusive, and the report never gives their status.** Line 13: "Its name, that the mask reaches the target another way, is no longer a live reading of this map."

- Amendment 2's account 2 is "the head's way may be open under the state's mask, with the gain carried by other subcomponents" (Amendment 2 report, line 91). Account 3 is that the mask carries what the head wrote. Both can hold: Γ may use the head's output to choose masks, and those masks may then let other subcomponents write B.
- The data lean that way. With the state's masks held and the head off in the forward, the masked gain keeps 12.2 to 12.4 of its 13.8 log-odds, on masks whose sampled slot reads 0.20 to 0.22. Line 125 says the same: "This is not 'the head computes B in the masked forward'".
- What the read removes is narrower: that the masks are independent of the head. Line 184's "not unrelated to the head" is the right size; line 13 is not.
- The plan's "A share near the target's means (c)" (plan line 439) sets (c) against (a) and (b). This run shows that contrast does not hold, and the report should say so rather than inherit it.

**Correction, line 13:** "Its name is no longer a fair summary: with the masks recomputed the gain does fall, so the masks are not independent of the head. The account behind it, that the gain is carried inside the masked forward by subcomponents other than those reading the head, is not removed. The frozen share still supports it."

**Add after line 127, the four accounts as Amendment 3 leaves them:**

| Amendment 2's account | status | evidence |
|---|---|---|
| 1. Masks close the head's way | at W₀ and at the state, unread beyond a 5% sample (F2) | sampled slot 0.008 at W₀, 0.20 to 0.22 at the copying states |
| 2. The gain goes elsewhere in the forward | live | frozen share 0.10 to 0.11 |
| 3. The head acts through the mask | supported as "masks react"; inheritance and off-distribution break not separated | recomputed share 0.61 to 0.69, about five sixths of it needing the masks to change |
| 4. The fine-tuned head is not in the forward | untouched | no read |

**F2. The slot read covers about 5% of head 4's slot, so "W₀'s mask shuts this head" is not shown.** Lines 11 ("already shuts that head out"), 29 ("how much of each head's output ... pass"), 39, 56, 72 ("does not run this head at this marker"), 184, 199.

Row 3 averages the gate over alive subcomponents with at least 0.5 of their read vector's squared norm in the slot. I read V and U for `h.2.attn.o_proj` from the checkpoint and reproduced the sets exactly (9, 67, 45, 25, 8, 23, so `slot_reads` is right). Then I asked how much of each slot those sets hold, over that matrix's 524 alive subcomponents:

| slot | (2,0) | (2,1) | (2,2) | (2,3) | (2,4) | (2,5) |
|---|---|---|---|---|---|---|
| subcomponents in the set | 9 | 67 | 45 | 25 | **8** | 23 |
| set's share of the slot's read norm² | 0.109 | 0.347 | 0.266 | 0.211 | **0.054** | 0.268 |
| set's share of $\sum_c \lVert U_c\rVert^2 \lVert V_{slot,c}\rVert^2$ | 0.113 | 0.334 | 0.320 | 0.242 | **0.054** | 0.248 |
| alive with at least 0.25 of norm² in the slot | 28 | 152 | 181 | 110 | **163** | 50 |

- 347 of the 524 have no majority slot. Head 4's weight is spread thinly: the largest single share of its energy is 1.6%, and that subcomponent has only 0.11 of its own read norm in the slot.
- The other 95% may be shut too. Their gates are not in `amend3.json`, so this is unread, not contradicted. The report mentions the exclusion at lines 49 and 194 and never its size, and head 4's coverage is the smallest of the six.
- So the table at lines 41-48 compares columns of unequal coverage. The "summed" rows also scale with how many subcomponents pass the threshold. "(2,0), (2,4) and (2,5) hold almost nothing" (line 52) and "(2,3) carries over three times the summed mask" are not statements about the heads' output.
- "Opens 25-fold" is true of the sample only.
- **The measurement that stands** is W₀'s frozen fall: 0.04 log-odds against 1.16, with the head off at the masks read with every head on. That fact needs no slot read.

**Corrections.**

- Line 11: replace with "And before any fine-tune, switching the head off in the unchanged model's masked forward barely moves it (0.04 log-odds against 1.16), which remains unexplained; a sample of the subcomponents reading that head reads nearly shut."
- Line 29: "reads the mean mask on the few subcomponents that read mostly one head's slot".
- Line 39: state "eight subcomponents holding 5.4% of the slot's read weight".
- Line 72: keep **Blocked**, reason "not read", not "W₀'s masked forward does not run this head".
- Lines 184, 199: the same.
- Line 52: drop the summed comparison across heads, or report the 5.4% to 34.7% coverage beside it.

**The same gap runs through the next steps.**

- What next 3 and Not run line 256 re-open "the eight subcomponents" and call that "the read for the unexplained anomaly". Opening 5% of the slot has little power, so a lean that does not return would say almost nothing. Replace with a weighted read: per window, the slot's output through the masked `o_proj`, $\sum_c g_c U_c V_{slot,c}^\top$ applied to the head's output, against every mask at 1, W₀ and each state. Then re-open all alive subcomponents by slot share, for instance everything at 0.1 or over (309), with a reference slot.
- This is $0, and it needs the gates, so one `debug` job.

## Findings that change a claim

**C1. The opening says "more", and the body says not to read it as "more".** Line 7: "fell by more than the model's own gain did". Lines 71 and 142: "Not read as 'more than'", "at least as much". The opening is the part the user reads. **Correction:** "fell by at least as much as the model's own gain did".

Line 11 has a similar slip: "when the masks are read with the head off" omits that the head is also off in the masked forward. Add "and in the masked forward".

**C2. "A few masks must matter a great deal" does not follow.** Lines 115, 205.

- `gates_moved` is a mean over the alive subcomponents of all 24 matrices (9,881, from `alive.pt`), at the marker only.
- 0.03 × 9,881 is about 290 gate units of absolute change at the marker alone, against about 28 at W₀. Most gates are small, since the slot means above are 0.002 to 0.22. So 0.03 may be a large part of the whole mask.
- Masks at the other 40 positions also move and also enter the forward, and `moved` does not see them.
- The size of the W₀-to-state mask change, which delivers the whole gain, is not in the same units.

**Correction:** "A mean change of about 0.03 sums to about 290 over the alive subcomponents at the marker; how concentrated it is, and how it compares with the change from W₀'s mask to the state's, was not read." Drop the Surprises item at line 205.

**C3. Two causes are stated without a read.**

- Line 85: "Head off still moves $L_{00}$ through Γ ... because Γ reads the target-model pass, where the head is working." The read shows the gate change moved $L_{00}$ by 0.35. Head (3,5), which also moves it by 0.21, shows some of this is not about that head working, as line 78 notes. **Correction:** "because Γ reads the target-model pass, which the hook edits; line 78's (3,5) reading says part of the change is not specific to this head."
- Line 121: "Closing a head's slot in that pass closes whichever masks Γ computes from it. That follows from what Γ reads". The hook replaces an output with a mean, it does not close anything. Γ is a learned transformer. On the lookup the same edit raises the gain. Line 194 admits this is an inference. **Correction:** "Replacing a head's output in that pass can move any mask Γ computes from it, whether or not the subcomponents involved carry the behaviour."

Line 123's "block 3's inputs change accordingly" was measured at W₀ on one batch, not at a state. Say so.

**C4. The lookup's slot is called shut, while it opens more than five-fold.** Line 132: "on the lookup the head's slot is shut at the state". The sample reads 0.045 to 0.059 against W₀'s 0.008, a factor of 5.4 to 7.2, and line 240 already says the 0.1 bar misbehaves. It also matters for the reading: the sampled slot opens several-fold on a map whose target share is about 0 and whose recomputed share is about 0. So a slot opening does not imply dependence on the head. **Correction:** "on the lookup the sampled slot opens less, 0.05 against 0.21", and add that sentence to line 39's reading.

**C5. "Not run" does not hold in three places.** Line 249 on.

- Amendment 2's "how much of that slot's weight passes under W₀'s mask and under each state's mask" is dropped. The plan (line 561) ran "a version" of it, unweighted, and the weighted one is what F2 needs. Add it first.
- Dropped from the plan's quoted list (lines 556, 558): "Which subcomponents the copying state's mask changes", "the lookup state's mask read for B; the copying state's mask read for the lookup's word". Line 258 is a different item, the masks moved by the head off.
- Not named anywhere: ablation with the state's own target-model mean. Every read here replaces a fine-tuned head by W₀'s mean, in Γ's pass as well. Line 155's "own-mask mean" is about the masked forward.

**C6. The plan's tree had a next step the report leaves unstated.** Plan line 341: head-dependent means "then the Qwen table". Lines 170-172 and 213 narrow the claim but never say whether the Qwen table is licensed. On this report's reasoning it is not. **Correction:** add to line 172: "So the tree's next step, the Qwen table, does not follow from this verdict either."

## Minor

- **Line 153.** With W₀'s fall at 1.16 the labels are within, over, within, so the map's verdict stays `head_more_than_target`, by 0.0015 on seed 1. Say the verdict's name survives while its margin does not. Line 142 also quotes one sd; the excess beyond the tolerance is 0.034, 0.057, 0.049, against a combined row sd of about 0.03 with the target's.
- **Line 29, "Each reference rules out one alternative".** Lines 129-134 say this is weak. Say "tests". Add the bars' power:
  - 0.15 of the mean gain is 2.07 log-odds on the copying map, which is more than the target model's own W₀ fall of 1.16. So that W₀ reference could hardly fire. On the lookup it is 0.92.
  - W₀ was read at a mask change a tenth of the state's (0.0029 against 0.027 to 0.030), so W₀ does not bound a break at the states. That fits line 132's scaling worry and is not mentioned there.
- **Line 33.** The filing says closed near 0.03 "on account (a)... No expected value on the other accounts". The report drops the condition, and the scorecard (line 229) reads "as expected".
- **Lines 54-58, "What is explained".** Line 39 and lines 61-62 say nothing shows that the closure causes this. Rename it "What fits".
- **Line 52.** (2,3) is not an arbitrary head. It is W₀'s second-ranked prefix head (`run.json` `stage_a.by_score_order`, 0.051, under the bar of 0.077), and its target share was indeed never read: it is not among the random draws. Worth one clause, as it raises the priority of line 260.
- **Line 162, "at most 0.04 log-odds under either real mask".** True of the copying map (at most 0.039). The lookup reads 0.125 to 0.373. Scope it.
- **Line 182, "reuse ... stays favoured".** Put E62's filed `copy_not_elicitation` (labels old, old, between) beside it, as the earlier check's C3 asked of Amendment 2.
- **"Lean" has two senses**: a weak prediction (line 33) and intact-over-broken log-odds, used at line 39 before its definition at line 57.
- **Row sd.** The bootstrap resamples windows as independent, and they share 32 keys. The filed rule uses it as is; the seed range of 0.08 is the better guide, as line 189 implies.
- **Provenance.** `amend3.py` records `git_head` but no dirty-tree status, unlike `run.json`. The diff from `6658095` to `f92b3ec` is empty for the code, and the zero-difference re-read covers the frozen path only. `gates_off` was in the commit, so the risk is low.

## What I checked and found right

- **Every number in the report's text and tables** recomputed from `amend3.json`, `amend2.json` and `run.json`: shares, gains, own falls, excess splits, W₀ falls, slot table and sums, sensitivity columns, $U_0$ end, probabilities, ratios, five sixths. All match to the stated rounding.
- **Filed rules applied independently of `copying.py`:** labels `over` ×3 and `within` ×3; ten references, largest 0.061, no `floor_high`; `head_more_than_target`; `sd_weak` false on all six. Predictions sha256 `94b13dd0...` equals the recorded value; the file has one commit, before the run's head.
- **Code.** Read critically, no bug found:
  - The hook sits on `model.target_model`'s o_proj (`ST.Net.net`), and row 1 tests it.
  - State gates are read under `model_state` with $W_0 + \delta W$ and the hook on; Γ is not edited.
  - The masked forward runs outside `model_state`, so with original subcomponents and the same `ci_len`, batch size and hook.
  - Frozen and recomputed differ only in the gates passed. The W₀ cells take gates read under the same head.
  - `w0_shift` and the references follow the filing. `V` is the read side, (d_in, C), and the view into slots matches `ablated`'s slicing.
- **Row 1** as filed: slot holds the mean, upstream 0, downstream at least 1.69, frozen differences 0, zero gain 0, gates moved.
- **The main judgement.** Keeping "masks follow the head" apart from "old mechanism" is right, as are inheritance and off-distribution as live alternatives, and "at least as much" as the reading of `over`. "No branch for this verdict" is disclosed correctly at lines 170 and 238, and `gamma_reading` groups the two readings, so the nearest branch is a fair call. Unfiled quantities are disclosed at line 247.
- **Figures.** `amend3_share.png`, `amend3_excess.png` and `amend3_slot.png` viewed; their values match the tables. The slot figure's title, "Which heads' outputs does each mask pass", needs F2's correction: "Mean mask on the few subcomponents reading mostly each head's slot".

## What I could not check

- Gates per subcomponent are not saved, so I cannot say what the other 95% of head 4's slot does under any mask. F2 is a coverage finding, not a claim that the slot is open.
- The energy measure ignores overlap between subcomponents and the input's statistics. It is a size guide, and the read-norm share agrees with it.
- I reran nothing on a GPU, so the results files are taken as the job wrote them; the log's verdict line agrees.
- `amend3_w0.png` and the notebook's Amendment 3 section were not checked, and the notebook likely repeats lines 11 and 13's wording. "Largest deficit of ten maps" (line 163) and McGrath et al. 2023 were not traced to their sources.
