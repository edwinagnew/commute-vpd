# Independent check of the E55 C3 step-ladder analysis

Reviewer: a fresh agent that wrote neither the code nor the report. Scope: does
`reports/analyses/2026-10-31_analysis_e55-c3-ladder.md` state correctly what the ladder found, score each filed
prediction by its filed rule, and draw only conclusions the data supports. Every headline number was recomputed with
`jq`/`awk` from `experiments/open_pair_instruments/results/ladder.json` (at 8a90db0) and the filed rules in
`predictions_ladder.json` (sha256 b39587b7…, equal to the run header's); the cipher's C3 held-out probabilities were
read from `cluster/logs/commute-e55-c3-seeds-46815.out`. `check.py` was run on the report (style flags only, 56
sentences; not reviewed further).

**Verdict.** No wrong numbers and no wrong verdicts. Every scorecard row follows its filed rule. Four statements are
misleading: VPD is credited on the same half-learned-cipher states the report says confound the cheap instruments; the
half-learned cipher is stated as the cause of C3's gap in "What we learned" although the report itself says no cause
was tested; the part A per-K bars are said to rest on a floor that never binds there; and "What next" sends the
comparison to the counting set without C1b's finding that the set has no headroom. The rest is minor.

## What was recomputed and holds

- **Controls.** `zero_D` = 0; 96/96 re-read checks true; 24/24 step-10 checks true; `w0_restored` true on all 36;
  no void state; no resume; git status clean at 1e2c122.
- **Part A.** Per-K means and sds of Δ_K/D per map match the table to 3 decimals. R gap 0.1464, noise 0.0508,
  bar 0.152: "no gap" (PK not met). Gap 0.191 at K 320 > 0.069 at K 40 (PK_trend met). Best K: rank-1 and elicit 320 on
  all 16; cipher 160 on 7, 40 on 1.
- **(a) Step ladder, R.** Step 10: +0.093 vs 3 × 0.0412 = 0.123 (noise is the cipher's sd; floor 0.029), no gap.
  Step 30: −0.129 vs 3 × 0.0346 = 0.104 (floor binds; pole sds 0.022 and 0.025), **reversed** by the filed rule.
  Steps 100 and 300: −0.112 vs 0.136 and −0.076 vs 0.128 (floor binds), no gap. PR 10/30/100/300 all "not met" as
  scored. `pass_step` null for both instruments, so Ppass met.
- **t½.** Gaps +2.62, +2.83, +1.98, +1.72, all the wrong way, all under the 3-step bar set by the 1-step floor at every
  step. Pole sds 0.47, 0.71, 0.71, 0.23, so 5.6, 4.0, 2.8, 7.5 pole sd; floor/spread 1.4 to 4.4. Every rank-1 seed's
  t½ exceeds every cipher seed's at every step (checked per state). Pt_reversed_10 not met: on the poles' spread the bar
  would be 1.41 and it would have read reversed, as line 293 says.
- **(b) numbers.** Ladder cipher held-out P(value) at step 10: 0.576, 0.544, 0.685 (mean 0.602); rank-1 0.976, 0.957,
  0.932 (0.955); elicit 0.979. All maps ≥ 0.993 from step 30. C3's cipher `p_o` at its scored checkpoint (log 46815):
  s0 0.450 (step 9), s3 0.390, s4 0.361, s5 0.360 (step 9), s6 0.318, s7 0.441: 0.32 to 0.45 as stated. `ladder.py:251`
  stores `ft.evals[t]["p_o"]`, the same quantity as the log's checkpoint `p_o`.
- **(c).** t½ falls 45% (rank-1 13.95 → 7.71), 47% (cipher), 47% (elicit) from step 10 to 300. Pearson r(norm, t½)
  over 36 states = −0.899. Within single steps r is −0.78, −0.95, −0.90, −0.89, but each step's 9 points are three map
  clusters, so this is the cipher-has-the-largest-norm effect, not a seed-level relation; the report's "untested" label
  in the cause table is right.
- **(d).** The "no gap" t½ verdicts are the filed reading and are labelled as floor-bound (lines 135-137, 243-246,
  293-294). The "clear ordering against the filed direction" gloss is supported: every seed is ordered the wrong way at
  every step. Correct as written.
- **(e) retention.** "Retention, not reuse" is labelled a candidate and untested at line 216, in "Not run" (line 320),
  conditionally in "What next" item 3, in the TRIED row and in the notebook. No place states it as found.
- Smaller figures: D_top 5.85 to 12.34 on every state, positive; |t_top − t½| ≤ 1.05 (rank-1 s0 step 30); every reverse
  halves by 17.97 steps; random removal at most 0.088 of D on any state at any K; D pole gap "ok" at 30 and 300 only;
  t½/D "no gap" at 10, "reversed" at 30-300; elicit positions all > 1.

## Problems

1. **Lines 228-230, misleading.** "VPD's two readings separated the same two maps by 7 to 10 times their bar in C3, at
   10 steps only. That makes VPD the only method of the three that reads the known order on these maps." The 7-10×
   checks out (C3: r 0.858 vs 0.235, bar 3 × 0.021; stand-in ratio 1.130 vs 0.327, bar 3 × 0.036). But C3's VPD
   readings were taken on the same cipher states the report calls half-learned (P 0.32-0.45) and identifies as the
   confound of C3's comparison (lines 171-172, 215, 225). The confound applies to VPD's separation exactly as it does
   to pruning's step-10 gap: VPD's cipher r of 0.235 is read on a fine-tune that had not yet raised its value above
   P 0.5. VPD at steps 30-300 is in "Not run" (line 319), so whether VPD keeps the known order once the cipher has
   learned is unknown. **Fix:** say VPD's C3 separation is on the same half-learned cipher, so it is subject to the
   same confound and untested on learned states; drop "the only method of the three that reads the known order on
   these maps" or limit it to "on C3's states, where the cipher was half-learned". The notebook and TRIED row
   ("VPD's own separation there is C3's, at 10 steps") should carry the same qualifier.

2. **Lines 224-225, misleading (also line 215's last clause, the TRIED row, PENDING line 11 and the notebook).** "C3's
   small gap in the filed direction came from comparing a learned old map with a half-learned new one" is stated as a
   cause. The report's own evidence section says "No cause is tested by a run. The half-learned cipher is read from the
   fine-tune logs, not varied" (line 251). Between step 10 and 30 the cipher's held-out P, its D (6.3 → 11.8 nats), its
   norm (3.5 → 5.8) and the step all change together, so learnedness is not separated from the growth of D or the
   norm; the evidence is three points of a trend (C3 states gap +0.146 at P ≈ 0.4, step 10 +0.093 at P 0.60, step 30
   −0.129 at P > 0.99) on 3 seeds. **Fix:** "C3's small gap in the filed direction appears only while the cipher is
   half-learned; it is consistent with that confound, which no run varied apart from D and norm". Line 215: "is
   consistent with the step-10 gap's direction" instead of "explains".

3. **Lines 108-111 (and the ladder_k legend "per-K floor taken over all K", and the TRIED row "unfiled all-K-floor bar
   0.189"), misleading.** The paragraph says the unfiled per-K statuses "use the same noise floor as the filed reading,
   the largest random spread over all five K, not each K's own. On that floor the gap at 320 neurons alone clears its
   bar". In part A that floor is 0.0340 (max random spread over the 16 pole states and all K), and it binds at no K:
   each per-K noise (0.0495, 0.0467, 0.0519, 0.0562, 0.0630) is the cipher's own seed sd at that K, and the filed R
   noise 0.0508 is the cipher's R sd. So the K 320 bar of 0.189 is set by the cipher's spread at 320, not by any floor
   choice, and the "ok" at 320 does not depend on the floor convention. The caveat that matters is only that the
   statuses are unfiled and that 320 is the edge of the range. **Fix:** "Each per-K bar is three times the cipher's seed
   sd at that K; the all-K floor (0.034) binds at none."

4. **Lines 269-271 ("What next" item 2), misleading.** "C2 on the counting set is where the brief's comparison can
   still be made: there the cheap instruments pass their positive control, so VPD meets a working rival." C1b's own
   report (`reports/analyses/2026-10-31_analysis_e55-c1b.md`, lines 13 and 18-21) found that an answer-only probe and
   the weight-change size within a recipe also score perfectly there: "The benchmark has no headroom left to rank VPD
   above these methods." A tie on C2 would then not distinguish VPD from trivial signals, and "If VPD fails there, that
   is the sceptic's result; if it ties, the brief's anchor item is met" reads past that. **Fix:** name C1b's
   no-headroom finding and say what a tie on C2 can and cannot show, or point to how C2 restores headroom if its plan
   does.

5. **Lines 232-233, minor.** "a noise bar from the poles alone": the bars were set by the 1-step floor at every t½
   step and by the random-removal floor at R steps 30-300 (7 of 8 cells); only R at step 10 used a pole spread. What
   changed from C3 is that elicit's spread was dropped. It cannot change a verdict (all t½ gaps are the wrong way), but
   the sentence overstates what was varied. **Fix:** "a noise bar without elicit's spread (floors bound in 7 of 8
   cells)".

6. **Lines 173-174, minor.** "The cipher's update is the largest at every step … so it also halves fastest" states
   the norm as the cause; line 218 correctly lists it as an untested candidate. **Fix:** "and it also halves fastest".

7. **Lines 168-169, minor.** "the cipher has not learned its map at step 10": all three cipher seeds (0.54, 0.58, 0.68)
   clear the project's own learned threshold `keydep.P_LEARN = 0.5` (`src/commute/keydep.py:27`), while all six C3
   cipher states (0.32-0.45) fall under it. "Half-learned" (used elsewhere) is accurate; "has not learned" is not by
   that rule. **Fix:** "is only half-learned at step 10 (P 0.60, just over the 0.5 learned threshold; C3's cipher was
   under it)".

8. **Lines 133-134, minor.** "At steps 100 and 300 the gap stays the wrong way, under the bar" is right for the filed
   R. The unfiled per-K block at step 100 reads "reversed" at K 160 (−0.156 vs 3 × 0.045 = 0.136); step 30 reads
   reversed at K 13, 80, 160 and 320. One clause would make the per-K picture complete.

9. **Line 293, minor.** "PR 30 is a reversed prediction." PR 30 filed a 20% lean that R passes, so "not met" was the
   expected outcome at 80%. What is unexpected is the instrument's direction (the "Surprises" entry, lines 255-258,
   says this correctly). **Fix:** "PR 30 is not met, and the instrument read reversed, beyond its bar".

10. **Line 217, minor.** Missing full stop: "whatever answer it gives The top-guess target".

Nothing else found. The control table, both step tables, the elicit table, the size table and the scorecard match
`ladder.json`.

## Not checked

- The figures beyond reading them against the tables (`plot_ladder.py` not reviewed line by line).
- The code itself (covered by `reports/reviews/2026-10-31_review_e55-c3-ladder-code.md`); log 46954 beyond provenance.
- Whether the plan's expected values (e.g. random removal under 0.02 of D; observed up to 0.088 at large K) should
  have been flagged as a further surprise.
