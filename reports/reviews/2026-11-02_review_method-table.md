# Independent check of the method table

*2026-11-02. Fresh agent; wrote neither the table nor the report. Checked commit 2625af6: `experiments/method_table/{build.py,plot.py,results/table.json,figures/shares.png}`, `reports/analyses/2026-11-02_analysis_method-table.md` ("the report"), `docs/research_log/parts/sections/methodtable.html` ("the section"), the plan and `decisions.md` against the source JSON. Rebuild script: `~/commute-data/scratch/baseline-lead/table-check/rebuild.py`. No model, no run.*

**In plain words.** The table's numbers are right, it rebuilds byte for byte, and its failed controls are shown as failed. "The question is still open" is the right top line, and the five rows moved to "only by assumption" belong there. The real problem is what the table leaves out. The cheapest columns on the rank-1 pair, and the small-model maps where r does not follow the build, are outside it, so "r separates every pair it read" reads stronger than the repo's results allow. One note about the counting set is also false on one of its two rows.

## Findings that change a conclusion

**F1. The cheapest methods are missing, and they separate the pair r separates.**
- **Where.** `build.py:50-68` (methods), `build.py:441-446` (C3's columns), the report's line 54, the section's line 5.
- **What the source holds.** `experiments/open_pair_instruments/results/run.json` `.score.readings`: W₀'s prior (`prior`) reads −5.00 against −14.32, status ok. The distance `D` reads 8.47 against 4.76, status ok, gap 3.71 over a bar of 1.45. `mlp_share` also reads ok. C3 files these as its baselines (`reports/analyses/2026-10-31_analysis_e55-c3.md:65-66`, `:149-152`).
- **Why it matters.** The prior needs no fine-tune at all. `experiments/TRIED.md:130` already says one cut on W₀'s prior sorts all eight maps into r's bands, so on the 67M "VPD has not yet shown anything the logits do not" (Spearman 0.95, post hoc). The report's line 54 says each cheaper method read on more than one fails on at least one. That holds only for the columns the table chose. The report's Blind spot bullet and line 54 mention the prior as a caveat on r, never as a rival method. The section never mentions it (F7).
- **Fix.** Add `prior` and `D` as methods, and `mlp_share` if kept. Fill C3 from `run.json`, and E58's maps from `amend6.json` and `amend7.json` `.prior_beside_r`. Note that E62 matched the prior by design (`.score.prior_ratio` 1.32). Add one sentence to the top: on the small model, one cut on the prior sorts E58's eight maps into r's bands, and only E62's unscored pair has r apart at a matched prior.

**F2. "Every pair it read" is true of the rows chosen, and the given reason for leaving out E58 is wrong.**
- **Where.** `build.py:620` says E58 Amendments 2-7 have "no method but VPD's or no reading". The report's lines 9, 54 and 236 lean on that.
- **What the source holds.** `experiments/open_pair/results/amend3.json`, `amend4.json` and `amend7.json` `.reads` hold `S`, `norm`, `r`, `ratio` and `x` for each state. `amend6.json` holds them for the prior-matched cipher.
- **The pair r does not separate.** At step 30 in `amend7.json`, the class map reads r 0.23-0.25 against its shuffled twin's 0.27-0.28. Filed: their difference −0.034, `kind_only`. Carry over its null is 0.55-0.64 against 0.08-0.09. Steering is 0.62-0.70 against 0.72-0.75. `decisions.md:112` keeps this open until the class map's mechanism is measured, so it belongs in the table as an assumption row. It is the one pair where a difference built in, a rule against no rule, does not show in r.
- **Other pairs left out.**
  - Copy against the prior-matched cipher: r 0.77 against 0.37-0.38; steering 0.82-0.88 against 0.15-0.17.
  - The shuffled map: r 0.47.
  - C3's own copy against cipher, on the same states as the rank-1 row: r 0.84 against 0.24, steering 0.64 against 0.18, pruning 0.65 against 0.39, about 5.5 bars. C3's report says the instruments "do tell maps apart". The table keeps 16 of the 32 states, so the report's line 9 "did not survive the port" hides this. Under the filed pole rule that sentence is still correct.
- **Fix.** Add rows for E58 A4, A6 and A7, and for C3 copy against cipher, marked only by assumption. Rewrite `build.py:620`. Change line 54 to "r separates these pairs, and not the class map from its no-rule twin".

**F3. The note that the labels equal task P and task N is false on the C1 row.**
- **Where.** `build.py:246`, `:252`; the report's line 44 ("Every counting-set cell carries that note"); the section's line 10; the star and footnote in `plot.py` and the figure.
- **What the source holds.** In C1 there is no task P. Every state is LOW fine-tuned to answer `C b 40` with the count of b, except R (`reports/analyses/2026-10-30_analysis_e55-c1.md:30-31`). The label can flip within one run: `.truth.scored` in `c1.json` has `a2/M3/1000` existing and `a2/M3/10000` new.
- **Why it matters.** The note dismisses the one row where the task stays fixed and the label varies, which is what the plan wants from the harder counting test. On that row, with C1b's cut-offs, pruning and steering sort 26 of 26, described only (`c1b.json` `.c1_described.agreement`). The real caveat, already in `extra` at `build.py:227`, is that the label is tied to the recipe, since every reuse state was read at 1K or 2K steps.
- **Fix.** Replace the note on this row with "task fixed; label tied to steps". Drop its star. Have the harder test's `/design` read this row first. VPD's reuse side is unread here: C2 read only some of its new states.

**F4. E62 Amendment 2 removes size as a live reading of r's gap.**
- **Where.** `build.py:517`, the report's line 62 ("Mechanism, context and size all remain live readings"), line 91.
- **What the source holds.** `experiments/induction_cue/results/amend2.json` `.row7_size` scales each update to the other map's size. The gap reads 0.686 against 0.691, filed `size_does_not` (`reports/analyses/2026-11-02_analysis_e62-amend2-masked-head.md:211`). Those results landed 8 minutes before commit 2625af6, and the analysis 2 minutes after it.
- **What stands.** Size still separates the pair as a labeller, so the labeller cell stays. Only the claim about r's gap goes. The amendment also narrows r on this map to "the original subcomponents can write this target under some mask". `copy_not_elicitation` stands.
- **Fix.** Add an Amendment 2 pointer to the r cell. Say on line 62 that mechanism and context remain, and size does not.

## Findings that change a cell or a sentence

**F5. Steering's "no gap" against TRIED's "baselines separate the ends" is resolved wrongly.**
- **Where.** The report's line 85, `build.py:452`.
- **There is no conflict.** The status is right: `no gap`, gap 0.032, bar 0.067. But TRIED's "baselines" (`experiments/TRIED.md:118`) are MLP share, D and the prior, all status ok. Steering is listed "beside" them (`analysis_e55-c3.md:65`, `:153`). `reports/reviews/2026-10-31_review_e55-c3-check.md:67` lists the same columns. TRIED is correct as written, and so is the JSON. The Surprise describes a contradiction that does not exist.
- **The verdict word.** Steering reads "control failed" here, but the filed gap control is for "each new instrument", R and t_half (`predictions.json` `.positive_controls.gap`, `.predictions.P1`). Steering has no such control. The same `no gap` reads "fails" for the size column on the same row (`build.py:471`), and steering's own miss on the half-known row reads "fails". Steering also reads copy 0.64 against cipher 0.18 on these states, so the apparatus works.
- **Scope.** At 96 keys and step 30, `amend3.json` and `amend4.json` give steering rank-1 0.27-0.31 against cipher 0.07-0.11, a gap of 0.20 at unmatched size. The no-gap is specific to the matched 24-key states.
- **Fix.** Delete the Surprise, and point at F1's columns instead. Make steering's C3 cell "fails".

**F6. Some "separates" cells have no filed rule behind them and no "derived" mark.**
- **Counting probe, `build.py:196-199`.** Filed role: "task-identity reference (not scored; not a pass)". The table gives "separates" with `held: true`, and no control exists.
- **C3 probe, `build.py:452`.** Filed as "reference only ... not counted as an instrument", and `ref_probe.all` is false. Its noise is 0, so the bar reads 0.00 and "held" is vacuous.
- **Counting labeller, `build.py:201-209`.** The verdict and the "21 of 21" in the cell rest on the within-cell score, which the cell's own note calls post hoc, while `derived` is null. The filed one-cut figure is 15, just at the bar.
- **r on C3.** `positive_control.held: true` is the pair's own gap. The report's line 54 discloses this, but the JSON field does not.
- **Fix.** Add a verdict "reference" for the probe cells, and set `held` to null. Mark the labeller derived. The report's count of eighteen derived cells then becomes nineteen. `build.py:611-612`'s rule says "with its control" and does not test it. That is harmless while the list is empty, but make it test or change the words.

**F7. The log section drops the caveat the report keeps.**
- **Where.** `methodtable.html:5`.
- **What is missing.** The words "prior" and "distance" appear nowhere in the section, yet it says r "separates every pair it read". The user reads the log and chat. Hover text shows only the reading and the pointer, so the filed "distance and prior" verdict and the per-cell notes are lost.
- **Cut text.** `build.py:578` cuts sparsity cells mid-sentence on the page: "this test passed by none, of the".
- **Fix.** Add the F1 and F2 sentences. Give sparsity a short reading of its own. Add `filed_verdict` to the hover.

**F8. E62's row is a causal check whose reuse side did not settle.**
- **Where.** `build.py:525`, `:520-522`; the report's line 85 ("only cheap method with a clean separation on a causal row").
- **What the source holds.** `run.json` has `.score.copy_reading` "mixed", seed 2 "between", `pair_ignored` true. TRIED:134 says target-in-context and old mechanism are confounded by construction.
- **Why it matters.** "By causal check" is the right kind of marking, but the labeller's "separates" reads as a win against a standing truth, and the lookup side is the only one labelled on every seed.
- **Fix.** Add to `how` or `detail`: "copy side not settled as filed". Drop the Surprise's claim, or word it as "against a label that did not stand".

## Smaller

- **F9. Size is "not run" where a source has it.** `build.py:543` says no source reads the labeller on the copy row, but `experiments/baseline_weights/results/results.json` per-matrix `norm2` gives copy 0.68-0.69 and random 0.75-1.34 (root of the sum). `decisions.md:107` makes size a standing column. The ladder's cell at `build.py:501` admits the sizes are saved. Fill both.
- **F10. E51's weight statistics are a method left out.** Top-10 share, intruders and effective rank are filed on exactly the copy row's eight states. The rank "separates" 24.5 against 23.9 and follows size (TRIED:107). E50's restore share, a decomposition reading on the same states, is also absent. Add E51 as a column and list E50 under left out.
- **F11. A narrow bar decides two steering cells.** On lowercase the gap is 0.184 (`build.py:316`, derived), and in nats on the half-known pair it is 0.180 (filed). Both clear the seed ranges with no overlap and miss only the 0.2 minimum. The half-known cell's log-odds share also sits on the figure's axis beside nats shares for other rows. Say both things beside line 54 and the figure.
- **F12. E52's failed null has `held: null`.** P5 failed by 0.0014 (`build.py:283-284`). It is in the notes but not in the report's control column. Show it as failed; the verdict stands.
- **F13. Sparsity's "fails" per row is the cross-test rule.** On lowercase the row passes three of five numbers. The verdict comes from "a number needs all four tests", not from that row. The report says so at line 85; the cell should too.
- **F14. Marking wording.** Six rows moved, not five: `e54_own` has `changed: False` at `build.py:434` with "as the row above". `ASSUMPTION` (`build.py:71`) cites row 112, but the label rule is `decisions.md:114`. "Built to elicit" does not fit rank-1, whose "old by construction" is the option (i) that `decisions.md:112` rejects, which is the stronger reason to move it.
- **F15. Order of next steps.** The report's lines 89-90 put Stage B first. The plan's line 13 and `decisions.md:117` put the harder test first, and the plan's line 29 calls Stage B "first candidate after the harder test". Reorder, or state the change.
- **F16. A pointer outside the repo.** `build.py:40` reads C2 from `~/commute-data/scratch/`, so VPD's only counting reading is not committed. At 89 KB it fits the results rule; copy it into `experiments/pcfg_anchors/results/`.

## What I confirmed

- **Reproduction.** `build.py` run into scratch gives a `table.json` and a section identical to the commit. There are 565 pointers, 533 unique, and all resolve; 62 of 108 cells are not run; 18 are marked derived.
- **Top line.** "Still open" is right. No row with truth beyond assumption has r standing: on C1b its control failed, on C1 its reuse side was never read, on E62 it is unscored. The derived list is empty by my own read too. F1 and F2 sharpen the line; they do not change it.
- **Verdicts from filed rules, on the cells marked "separates".**
  - C1b steering, pruning and reverse: 21 of 21, `passes` true, controls true in `c1b.json`.
  - E54 r: a_higher, gap 0.518, range 0.055.
  - E46 steering: P3 pass, gap 0.628.
  - E54 own-step steering: P3-own a_higher, and "not a test" is kept.
  - C3 r and ratio: status ok.
  - E51's rule recomputed on all eleven derived cells that use it.
  - C1's "fails" follows the filed every-state rule, and the per-run passes are kept in the filed column.
- **Failed controls shown as failed.** C2's identity prompts read 0.07 on r and on ratio against 0.5. Pruning and reverse on C3 and at every ladder step (`pass_step` null, K re-read `no gap`, step 30 `reversed`), with the planted edit's pass shown beside them. Readouts: fit bar missed; construction check 11 against 24.
- **Caveats that stay attached.** E63's low power is in each cell's notes and in the report and section prose, and its wording matches the E63 report. The test mapping is right: T3_C W2, A1, C1; T1 A1; T2 W2; T3_B none. E64's branch is verbatim, with P1 pass, P2 fail, P3 pass. E62's `r_scored` false and verdict are kept. The counting note is right on C1b.
- **Truth markings.** C1b by construction agrees with `decisions.md:114`, and `all_disagreements` is empty. C1 by causal check agrees with its source. The five moved rows are right: E60's labels read "none yet", E64 says "no causal label", E52's amendment says "mechanism claim only", and I found no causal check on their states in TRIED's rows or in the E54 report. No state changes side.
- **Plan scope.** All named sources are present, one results file and one log section, and the failed ports are shown.
- **Writeup check.** `check.py` on the report: 0 errors.

## Not run

- Every cell by hand. I sampled every source, and all "separates" and derived cells; the rest rest on the rebuild.
- E57, E34 and the Qwen line beyond their TRIED rows. They have no truth label and no cheap method beyond size and random changes, so leaving them out looks right.
- Any prior or D reading on the E46, E53 or E54 rows. Whether they tie r there is unmeasured from these files.
- The figure's numbers beyond sight against `table.json`.
