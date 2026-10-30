# Review: independent check of the E55 C3 analysis report

2026-10-31 (written 2026-10-30 UTC), reuse line. Reviewer: a fresh agent that wrote neither the code nor the report.
Checked: `reports/analyses/2026-10-31_analysis_e55-c3.md` (at 514750b) against
`experiments/open_pair_instruments/results/{predictions,measure,seeds,run}.json`, the plan
`reports/plans/2026-10-31_plan_e55-stage-c-open-pair.md` (Amendment 1, Filing B note), `src/commute/revival_kv.py`
(`column_reading`, `tracks`, `verdict`, `joint_verdict`, `score_columns`), the three figures, and E58's Amendment 6
section. Every number below was recomputed with jq from `run.json` / `seeds.json`.

## Verdict in plain words

The numbers and the scorecard are right: every filed prediction is scored by its filed rule, and the headline (neither
instrument separates rank-1 from the cipher; r and the stand-in ratio read "distance and prior") stands. The problems
are in the explanation. The claim that both instruments "track D" rests on three map clusters and is contradicted by
the fourth map, copy, and by near-zero correlations within each map. One table column says it applies the filed rule
to the reported rows and does not. Two opening sentences overstate. The first refinement proposal cannot give the
verdict it says it could, because rules 5-6 decide it in advance on this axis.

## What checks out

- Gap, noise, status and position for every column (`run.json` `.score.readings`): R gap 0.0685 vs 3 x 0.0468 = 0.140;
  t_half gap +3.21 vs 3 x 1.423 = 4.27; r 9.8, ratio 7.4, x 4.5, MLP share 2.8, D 2.6, S 0.47, norm 0.13 bars. Positions
  and 90% intervals as in the table at lines 140-152.
- Verdicts: R and t_half "no gap"; r and ratio "distance and prior"; joint "not settled". Scorecard leans match filing B
  (P1 55/65%, P2 bucket 50%, P3 bucket 35%, V not settled 85%, C0b 85%, P_copy midpoint 0.791 filed).
- The unfiled 0.1-only column: r is 0.063 from p_prior (prior), ratio 0.148 from p_prior and 0.66 from p_D (VPD), x VPD,
  MLP share 0.030 from prior (prior), q 0.070 from p_D (distance). Correct.
- Per-K: R gap 0.006 at K 13, 0.069 at K 40; K 40 better on 31 of 32 states. Half-life over D 1.58 / 2.14, 2.3 x the
  largest sd, elicit 1.03, copy 1.17. Outlier-dropped gap 0.083. 3.6 standard errors. Random 40-unit removal 0.003-0.007
  of D. Carry labels (rank-1 neither 7 / new 1, copy neither 6 / reuses 2, elicit and cipher new 8). Scored steps 10, and
  8 or 9 for the cipher. Probe per block. Planted control (12 of 13, R 0.973, t_half 3.97).

## Problems

1. **Lines 144-146 and 154: the "filed reading" column does not apply the filed rule. Wrong.** Line 154 says "the other
   rows apply the same rule for reference", but carry x, MLP share and probe q are listed as "VPD side". Under rules
   5-6 (`revival_kv.tracks`: same firm side as a firm baseline) each of them reads "distance and prior": x's firm side
   is VPD (5th percentile 0.693 >= 2/3), the same as p_D's and p_prior's. Fix: change those three cells to "distance
   and prior", or retitle the column "firm side, before rules 5-6" and drop the "same rule" sentence. It matters for
   (c) below, because the carry's old-end reading is then no more informative than the prior's on this axis.

2. **Lines 13, 168, 171-172, 181 and Surprises: "both instruments track D" and "the map adds little" are not
   supported. Misleading.** The half-life to D correlation of 0.92 is between three map clusters. Within each map it is
   0.44 (rank-1), -0.10 (cipher) and 0.48 (elicit), and with copy added it falls to 0.71. Copy sits at D 7.8 but halves
   in 9.1 steps, faster than the cipher at D 4.8. On pruning, copy gives back 0.65 of D against the cipher's 0.39, a
   difference of 5.5 times the bar, at a lower D than rank-1. So the instruments do tell maps apart. They just do not
   order rank-1 against the cipher in the filed direction. Three clusters cannot separate "D drives the reading" from
   "the map drives both". Line 172 "that is why reverse fine-tuning leans the wrong way" names a cause that no run
   moved, which `.claude/CLAUDE.md` forbids. Line 243 itself says the cause was not tested. Fix: "the half-life orders
   the three scored maps as D does (0.92 across states, between maps only; within maps -0.10 to 0.48); copy is off that
   line", give copy's R and half-life as evidence that the instruments do read the map, and write "consistent with"
   in place of "that is why".

3. **Line 181: "for pruning, normalising by D removes it" is contradicted by the data. Misleading.** Across the 24
   scored states R still correlates 0.80 with D, and the map means run in D order (cipher 0.39 at D 4.8, rank-1 0.46 at
   8.5, elicit 0.51 at 10.1). The filed-direction R gap could itself be partly D. The basis is the same three-cluster
   one as problem 2. Fix: state the 0.80, and say the R ordering is no less D-ordered than the half-life.

4. **Line 120 and lines 110-111 / 247-248: the t_half bar is set by elicit's spread, which the report does not say.
   Misleading.** The noise is elicit's sd of 1.42. Rank-1's is 0.75 and the cipher's 0.78. The verdict "no gap" is
   right under the filed rule, which takes the largest sd over all three maps. But the poles differ the wrong way by 4.1
   times the larger pole sd, about 8 standard errors of the mean. "Under the bar, so not a reversal" undersells a
   reasonably clear wrong-way ordering of the poles. Fix: one sentence: "the bar comes from elicit's spread, 1.42; on
   the poles' own spread, about 0.78, the wrong-way lean is about 4 sd".

5. **Line 14: "Only the decomposition separates the two ends on these maps" is false as written. Misleading.** Line
   131 and the table show that D, the prior, MLP share, the probe and behaviour's graded carry x all separate them.
   Fix: "Of the two methods meant to read mechanism without a decomposition, neither separates the ends. VPD does, and
   so do the baselines and the graded carry."

6. **Lines 226-228: "fail their positive control on real poles, which is a statement about the port, not about the
   methods" goes past the data. Misleading.** Nothing here tells "the port" apart from "these methods cannot see this
   difference in a 10-step update". Line 243 says so ("no run varied the update size, the target or K"), and proposal 1
   (line 262) treats the question as open ("if they still cannot, the port is the problem"). Fix: "this does not yet
   say whether the port, the update size or the methods are at fault".

7. **Lines 172-174 and 180: "thin layer ... yes every time" does not explain the gap between the planted and real
   states. Minor.** The planted wrapper is also a small edit (alpha 0.59 on 13 columns), and it is revived at 0.97 and
   halved in 4 steps. What separates it from the real states is concentration (13 units, one block) against spread
   (about 30% attention, more than 40 units). Real states get back 39-65%, not "yes". The K trend, with the gap 10
   times larger at 40 than at 13, points at spread. Fix: name spread over units as the leading candidate beside update
   size, and soften "yes every time" to "about half every time".

8. **Line 180: "largest per-matrix relative change ... about 1.4%" has no source. Minor.** It is not in any results
   JSON, and line 325 says the extra numbers were computed from `run.json`, which holds no per-matrix W₀ norms. Fix: name
   how and where it was computed, or drop it. Also on line 180, "halves every map's change in 8.7-18.0 steps" is wrong for
   copy, whose minimum is 7.9. Say "the three scored maps" or "7.9-18.0".

9. **Line 242: "E58 Amendment 6 already showed r does not follow the prior" overstates E58. Misleading.** E58's
   Amendment 6 section says it rules out a smooth dependence on the prior, not a cut on it: "a cut on the prior anywhere
   between 3.7e-5 and 5.9e-5 still puts ... all six maps" in the right bands. Its rank correlation of r with log prior
   is 0.89. It was also run at 128 keys, step 30. Fix: "showed r is not a smooth function of the prior (a cipher at a
   prior 1.6 times below copy's read 0.37-0.38), though a cut on the prior still sorts every map".

10. **Line 250: "Elicit is the hardest map to undo, by both instruments in nats" holds only for residual nats.
    Minor.** Pruning gives back the most nats of elicit (5.12) and a higher share than of rank-1 or the cipher. The claim
    holds only for what is left after pruning (4.93 nats against rank-1's 4.56). Fix: say "the most left after pruning,
    and the slowest to halve".

11. **Line 261: typo "E12.s".** Minor.

## (c) The carry label, x at the old end on eight seeds

It does not remove the disagreement that motivated C3, but it moves where the disagreement sits, and the report does
not say this clearly enough. On three seeds x placed elicit at 0.69, with an interval straddling 2/3. On eight it sits
at 0.72 (0.693-0.756). That is a narrower interval around nearly the same point, not a new effect. The label is
unchanged: elicit "new" on 8 of 8 seeds. On the construction axis, though, behaviour's graded carry sides with VPD.
Elicit carries -0.036 and rank-1, old by construction, only 0.081. Both are near zero, against the cipher's -0.344. The
label also calls rank-1 "neither" on 7 seeds and "new" on 1. So E58's "VPD reads old, behaviour reads new" is, on this
axis, a disagreement between VPD and the label's fixed cut, and that cut does not read old against new even on the
map that is old by construction. That is the plan's outcome (e) reading about the label. Its formal trigger, the
instruments placing elicit with rank-1, did not fire, so it stays a reading, not a finding.

Two qualifications belong next to it. First, under the filed rules x reads "distance and prior" (problem 1). So on
this axis the graded carry, like r, cannot be told apart from the prior. Second, x is a different quantity from the
instruments: it measures carry, not possession. The report mentions the change twice in passing (lines 136 and 251).
Lines 26-28 still frame E58 as "VPD and behaviour disagree" without noting rank-1's near-zero carry, and "What we
learned" says nothing about it. Fix: one sentence in "What we learned": "On the rank-1-to-cipher axis the graded carry
puts elicit at the old end, 0.72 on eight seeds. The E58 disagreement is now between VPD and the label's cut, which
calls the old-by-construction rank-1 map 'neither' or 'new'. Under rules 5-6 this reading too is 'distance and
prior'." Add rank-1's carry of 0.08 at line 28.

## (d) The refinement proposals

- **Proposal 1 (longer fine-tunes, same axis) cannot give its stated verdict. Misleading.** Lines 269-270 say a gap
  with elicit at the old end would show the granularity limit is shared. The prior's position, 0.92, is a property of
  W₀ and the maps, so it does not change with the step, and it stays firm on the VPD side. Under rules 5-6 any
  old-end instrument reading on this axis therefore reads "distance and prior" at any step, and "shared" stays
  unreachable, as filing B noted. Longer runs will also widen the D differences that problem 2 says the half-life
  follows. Fix: pair proposal 1 with a change that frees the verdict from the prior. Options: a map that separates the
  prior from old pieces (copy, or E58 Amendment 6's prior-matched cipher, as a pole or placed map), or a re-filed rule
  6. Say which before claiming the conditional.
- **The cheapest lead in the data is missing.** The K trend, with the gap 0.006 at K 13 and 0.069 at K 40, is the one
  in-data hint for a refinement. Re-reading the 32 saved states at larger K is a $0 job of about a minute on the
  cluster. It is listed only under "Not run" (line 310). Fix: add it as proposal 0.
- **Proposal 2 (match D)** follows from the data, with problem 2's caveat that D alone does not explain copy.
  The half-life over D is fairly labelled unfiled.
- **Proposal 3 (target = log-probability of W₀'s top guess) is close to a prior column. Minor.** Rank-1 trains toward
  exactly that token, so its distance on this target is negative, and rank-1 sits at W₀ by construction. The control
  would then pass trivially, and the target would read possession of W₀'s guess, the thing rule 6 relabels. Fix: keep
  "W₀'s ranking of its top answers" and drop the top-guess variant, or say what the half-life means when the state is
  nearer than W₀.

## Not checked

- The seed-0 retrain "equal to 5 decimals" claim beyond the `checks` booleans in `seeds.json` (all true at the filed
  tolerances).
- The 1.4% per-matrix change, which would need W₀ and the state files loaded (problem 8).
- "Rank-1's 32 keys have only 18 distinct values".
- The notebook section and the research-log line.
- `check.py` was run and reports style flags only (sentence length, numbers per sentence), no retired terms.
