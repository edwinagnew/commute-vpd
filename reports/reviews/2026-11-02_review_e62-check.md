# Independent check of E62's analysis and notebook

*2026-11-02. Fresh agent; wrote neither the analysis nor the run. Checked `reports/analyses/2026-11-02_analysis_e62-copying-heads.md`
("the report") and `notebooks/E62_induction_cue.ipynb` against `experiments/induction_cue/results/run.json`, `predictions.json`, the job 48541 log,
the plan, `src/commute/copying.py`, `experiments/induction_cue/run.py`, `local_faith.route_cells` / `route_split`, and E58's `amend4/6/7.json`.
Recomputed by script: `~/commute-data/scratch/regating-lead/e62-check/recompute.py`, `states.py` (run through `cpu-run`). No model was loaded and no GPU job ran.*

**In plain words.** Every number I recomputed matches, and the filed rule was applied correctly, so "r not scored" stands. The problems are in what the report concludes around the rule: it says the prior account fails without the limits its own table shows, and it leaves out update size, which separates the two maps as cleanly as r does. No rerun is needed; four edits are, and the numbers do not change.

## Findings that change a conclusion

**F1. Update size is missing, and it is a third live reading of the r gap.** `decisions.md:107` makes each state's update size a standing column in every instrument comparison, and the plan repeats it (plan lines 35 and 161). The report has no such column, and the words "norm" and "||ΔW||" do not occur in it. `run.json` holds the sizes:

| | copy, seeds 0 / 1 / 2 | lookup, seeds 0 / 1 / 2 |
|---|---|---|
| ‖ΔW‖, step 30 | 4.62 / 4.57 / 4.62 | 5.45 / 5.49 / 5.50 |

Same recipe cell, a factor 1.19 apart, with no overlap: the size separates the maps as cleanly as r does. Over the report's ten maps at step 30 (E58's `reads.*@30.norm`), the Spearman correlation of mean size and mean r is about −0.52 by hand, and the three smallest updates are the three highest r: E62 copy, E58 copy, rank 1. The class maps are the counterexample, at size 5.2 to 5.3 and r 0.24 to 0.28. So report line 237, "leaves mechanism and context as live readings", is too narrow. Notebook line 211's "a lookup of about the same size" is unsupported by a number and is off by 19% (the gain is off by 21%). Fix: add the column to the r table, add size to line 237 and to the bearing table, and say what the notebook means by "same size". Size may only be a proxy for how easy a map is, and this run cannot tell.

**F2. "The prior account fails" is an unscored reading stated without its limits.** Report line 237 says "The prior account fails here" with no "unscored". Notebook line 198 opens "No:", and line 211 says "The prior alone does not set r". Lines 187 and 221 are hedged better. Three limits, the first two from the report's own data:
- The match holds on one measure only. On first training batches the ratio is about ×4 and on mean probability about ×25 (report lines 195-196; I recovered these from the log's step-1 `ce_o` and `p_o`).
- The match is in-sample. Each word was selected from 24,994 candidates on its key's 3 to 11 read windows (`copying.py:330`, `run.py:192`; window counts from `windows.pt`). On a training batch the lookup's prior falls to 1.1e-4 to 1.7e-4 from 2.39e-4, as selection would predict, though that is one batch and I cannot separate the two.
- The numbers match, but not their kind. B's 3.15e-4 is 65 times its own level with B absent (4.84e-6), so nearly all of it comes from context. A lookup word is out of its windows, so its 2.39e-4 is not lifted by its own presence. The report never sets these side by side, though it is the same confound line 171 raises.

E58's own pair (r 0.38 and 0.77 at a factor 1.6) already showed the prior does not fix r at a small gap, so E62 repeats that rather than newly refuting a prior account. Suggested wording: "unscored, and only on geometric-mean probability on the read windows, the two maps are close in prior and far apart in r".

**F3. The share has the same low-start issue the report finds in r, and it is not applied there.** The share is net of W₀'s own change (`copying.py:431`, `run.py:318`). If the head's removal lowers W₀ by δ, `G_abl` is measured from that lower start, and the state's own log-odds fall by `G − G_abl + δ`. Using the report's own bracket for δ (0 to 4.18), the state's fall as a share of the gain is:

| seed | filed share | state's own fall, at the bracket's far end |
|---|---|---|
| 0 | 0.563 | 0.894 |
| 1 | 0.545 | 0.877 |
| 2 | 0.471 | 0.804 |

The filed rule is unaffected and the miss stands. But "carries about half" (line 13), "×253 left" (table, line 41) and "partly" (line 241) all hold only relative to an ablated W₀ that was never read, and "the odds stay hundreds of times above" can be largely a lowered floor. `lo0_H` is computed at `run.py:311` and discarded. The report lists this read first under Not run but only as a reference for the bar and a number for the bracket; it should also say that "about half" could be a large underestimate of the head's effect on the behaviour.

**F4. The reading of head `(1,1)` assumes it acts beside `(2,4)`; the evidence points to in series.** Report line 233 says "Reuse may be larger than the share", line 80 says the other head "would explain why every seed sits near 0.5", and the notebook (line 84 onward) says part of the remainder "may run through a second head". But each head alone takes W₀'s top-1 from 0.82 to about 0.01, which is what two steps of one chain do, and is how Olsson et al. describe the circuit (the plan cites their Argument 5). If `(1,1)` feeds `(2,4)`, much of its 0.23 to 0.29 sits inside the share already and adds nothing to the remainder. Line 70's "shares need not add" is correct but points only at the ceiling. Fix: say the overlap could run to full, so `(1,1)` gives no evidence that reuse exceeds the share until both are read together. Remove "That would explain" from line 80.

## Findings that change a claim, not the conclusion

**F5. "The behaviour largely goes on any seed" and "at most 0.19" do not follow.** Report line 56; notebook line 84 ("to a value at most 0.19"). The bracket is the probability at the mean log-odds. It is not a bound on mean probability, and the report's own W₀ row shows how far the two can sit apart at low levels (geometric mean about 5e-4, mean about 1% on the first batches). The table at lines 50-54 also mixes two measures: the head-on column is mean probability (0.968, `p_read`), where the head-on value at the mean log-odds is 0.990. Line 56's caveat covers only the bracket columns. Fix: delete "at most" in the notebook, use one measure in the table, and weaken line 56 to "may largely go".

**F6. The most direct read of "runs through the head" is not in Not run.** Nothing in lines 298-312 reads `(2,4)`'s attention at the marker on the cue windows, under W₀ and under each state: does it attend to B, and does the fine-tune move its query? `attention_patterns` exists, and line 81 admits the prefix score is off-format. The blind-spots line (247) also omits downstream compensation after ablation (McGrath et al. 2023, the hydra effect), which would make the share an underestimate. Both belong on the lists.

**F7. "Only step-30 states were saved" overstates the barrier.** Lines 79 and 302. The path control read a repeat difference of exactly 0, so the fine-tune is deterministic and a 300-step fine-tune took about 12 s (log). Shares at steps 100 and 300 cost a short `debug` job. Worth saying, because of the explanation below.

**F8. A cause is named without a read.** Line 171: the lookup's word "must be a table entry the original subcomponents do not hold". Yet the same table has original subcomponents, under the new mask, raising that word by 5.7 to 6.4 log-odds, with no copying path. Either they hold part of the entry or the mask carries it (line 172's second reason). That rise is also the best available size for r on an arbitrary no-rule map: about 0.3 to 0.4 here and in E58's cipher runs, not 0. Change "must be" and say this.

## Minor

- **Notebook has no departures list.** Required by `.claude/skills/writeup/SKILL.md:100-103`; its mean ablation is described with no note that Olsson et al. zero-ablate. The report's list is fine.
- **Notebook line 99.** Seed 2, step 300 reads 0.64; it is 8.539 / 13.454 = 0.635, and the report's 0.63 rounds correctly.
- **"Not window sampling"** (line 76, notebook line 82: "seed 2's miss is a real difference"). The gap between seeds supports this. The miss itself is 1.8 row sds from the bar, and the bootstrap is per state and unpaired, on shared windows. Say the seeds differ; don't say the miss is beyond sampling.
- **Line 78's seed pattern is not monotone.** Seed 1 has the lowest broken ratio (0.336) but a lower share than seed 0 (0.396, 0.563). Unmentioned, and it weakens the suggestion. Also, "uses neither head" was not read.
- **Scorecard line 268.** `b_prior` filed "inside [1e-6, 1e-4]" and read 3.15e-4. That is a missed prediction (known before this run), and the row should say so, in line with the other misses.
- **Expected value unreported.** The plan's table expected the lookup's specific share at 0.65 to 0.72; it read 1.44 to 1.46, because the other words fall by about 7. Line 104 notes this inflation for the copy map only.
- **Line 102.** "Passes the bar" reads as success, but the bar is a ceiling (≤ 0.5). Say "goes over".
- **Line 221.** "Three no-rule lookups" leaves out E58's shuffle (each key takes another's value), at prior 1.0e-5 and r 0.47 to 0.48, above E62's lookup at 24 times the prior. Including it helps the sentence's point, but changes "a factor of 170 moves r by about 0.1" to a range of about 0.2 if class shuffled joins too.
- **`r_prior.png`.** E58's x values are `w0_geo_p_trained`, E62's are read windows. Not visible on this log axis, but the axis label implies one measure.
- **Line 181.** "The original subcomponents can already write this target under some mask" is close to the definition of a high L01. The content is in the preceding sentence.

## My own reading

1. **The head result is clean.** With `(2,4)` off, the copy map loses 0.47 to 0.56, the lookup 0.00, and most other heads about 0.01. That difference needs no decomposition. The bar's miss is 0.03 on one seed, and the rule rightly stands, but the head-level reading is firmer than `copy_not_elicitation` sounds.
2. **An explanation the report does not consider: the head may carry the pair-matched part, and the remainder may be the loose copying.** What survives with the head off tracks the gain with the pair broken: 5.53 beside 5.01, 5.75 beside 4.25, 6.64 beside 6.84. Seed 2's narrow miss then reflects only that it is loosest at step 30. Three seeds, seed 1 off by 1.5, so a hypothesis only. It predicts two things: the head off barely moves the gain on broken windows, and the share falls with steps, to about 0.37 to 0.42 by step 300 (one minus the broken ratio). The second would mean "reuses an old head" is a statement about step 30, and that more training builds a route around it. Both reads are $0.
3. **r high on the copy map is not yet evidence of mechanism.** Context, update size (F1) and a roughly 0.3 to 0.4 floor for arbitrary maps (F8) all sit beside it. L01 against L11 (a difference of 0.01 to 0.03) is striking, but a mask built from fine-tuned activations is not old weights, as the report says.
4. **What I checked and found weak: where the update sits.** A lookup could sit late, beyond the reach of masks, and a rule could sit upstream. Saved updates tilt only slightly: last-block MLP 21.6 to 21.9% of squared size on the lookup, against 16.7 to 17.3% on the copy map. Adam spreads updates by parameter count, so these norms are nearly blind to this. Not excluded; a gradient or mask-by-block read would be needed.

## Verified and found right

- Share, `G`, `G_abl`, lost, row sd, odds left, median other head: all six rows of lines 41-46.
- Probability bracket, lines 52-54, from −8.06 and −12.24 (recomputed from the geometric means).
- All five random draws' top-1 and shares, lines 64-68; "four to five times"; "at most 0.85".
- Seed-2 figures: 1.8 row sds, 0.09 range, step-100 ordering, prefix rises at most 0.037 and 0.017, top other head 0.065 at `(2,3)`.
- Pair-broken table, lines 91-100: every gain, ratio, log-odds, percentage and gap. W₀ factors 13.5 and 4.8; job 48507's values.
- Route cells and r, lines 128-133. Filed shares sum to 1.000 (−0.217 and 0.121 on seed 0). Alternative starts, lines 141-144.
- Ten-map table, lines 152-161, against E58's `reads.*@30.cells`, seed counts included; largest deficit claim right.
- Prior tables: 1.32, ×3.4, first-batch rows from the log, 3.15 over the band, factor 173, Spearman 0.95 on E58's eight maps, 0.45 to 0.51 on the copy map's gain.
- E57's random change at r about 0.99 (0.993 to 0.994, its analysis line 93).
- Advice values 0.75 and 0.7; `(1,1)` reasoning quoted fairly from the advice.
- Filed rule: `score` order at `copying.py:520-541` gives `copy_not_elicitation` and `r_scored` false; `pair_ignored` true from seed 2. No bar re-read after the run. r is labelled unscored in the headings.
- Target column unseen by any forward (`route_cells` `x = batch[:, :ci_len]`); L00 differs between maps only in the token read.
- `windows.pt`: A at p, B at p + 1, one marker, no word in its own key's windows (0 of 200 and 0 of 800), 32 distinct words, train and read Bs disjoint. Train and read rows are disjoint by the `held` mask; the stored row indices are relative to each subset, so their 9 equal values are not shared rows.
- Sha256 of predictions, windows and a state match; `predictions.json` committed before the job; job time 3 min 25 s; step-10 readings 0.87 to 0.90 and 0.54 to 0.57.
- Figures: `share.png`, `cells.png`, `r_prior.png` show what their captions say, including bars and the unscored marking.
- Granularity limit present in both files (report line 241); report's departures and Not-run lists present; `check.py` reads 0 errors on both.

## Not checked

Olsson et al. and Jain et al. against the papers themselves (taken from the plan's text). Line 17's claim that every earlier map had the two readings aligned. Token frequency ranks of the lookup's words (only ids read). The model was not loaded, so no ablated or masked number was re-derived, only the arithmetic on saved values.
