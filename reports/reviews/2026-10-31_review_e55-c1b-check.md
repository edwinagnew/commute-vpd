# Review: independent check of the E55 C1b analysis report

Reviewer: a fresh agent that wrote neither the C1b code nor the analysis, 2026-10-31. Object:
`reports/analyses/2026-10-31_analysis_e55-c1b.md`. Read against `experiments/pcfg_anchors/results/c1b.json`,
`c1b_cutoffs.json`, `predictions_c1b.json`, the plan `reports/plans/2026-10-30_plan_e55-c1b-refined-instruments.md`,
`experiments/pcfg_anchors/c1b.py`, `c1.py` (data builders), `src/commute/interchange.py`, `src/commute/pcfg.py`,
`plot_c1b.py`, the three `c1b_*.png` figures, the cluster logs of jobs 46330 to 46499 and the git history.

## Verdict

The report states the run correctly and scores every filed prediction by its filed rule. Both main claims hold.
(a) Pruning, reverse fine-tuning and steering each label 21 of 21 scored states and 12 of 12 per run against a
requirement of 17 and 10. The margins are real, no row is swap-built, and the cut-offs were committed before the first
scored reading. (b) The truth label coincides with the trained task. The answer-only probe and a per-recipe
update-size labeller both reach 21 of 21, so on this set and this scoring rule VPD can at best tie. I found one
factual error, two misleading passages and three minor points. None of them changes a verdict or a conclusion.

## What I recomputed (all match the report)

- **Scores.** From `.scores`, all three scored instruments have n 21, agree 21, bar 15, nuisance lr 12, steps 12,
  update size 15 (threshold 0.03345, small means existing), needs 17, passes true. p-values are 4.77e-7 (all) and
  2.44e-4 (per run). Per run: n 12, agree 12, bar 10, nuisance 6 / 6 / 9, needs 10. No state was void.
  `train_stops.needs_full` reads n 21, n_c 15, needs 17, impossible false.
- **Cut-offs.** Each one is the midpoint of the calibration gap in `c1b_cutoffs.json`. Pruning is
  (0.0820 + 0.36175)/2 = 0.2219, reverse (0.1304 + 0.4130)/2 = 0.2717, steering (0.6686 + 0.8156)/2 = 0.7421 and the
  probe 0.4735. Each has balanced accuracy 1.0 with 8 existing and 6 new.
- **Margins** (report lines 113-117). On the scored states, steering's nearest readings are 0.830 and 0.656 against
  0.742, i.e. +0.088 and -0.086. Pruning's are 0.602 and 0.080 against 0.222, i.e. +0.38 and -0.14. Reverse
  fine-tuning's are 300 and 950 steps against 0.2717 × 2,300 = 625 steps, 325 steps either side.
- **Per-state table** (lines 193-215). Every row matches `.scored` and `.nuisance`: update size, `prune.acc0`, pruning
  gain, reverse steps, S, block 5 and the labels. G_score is 10.48-10.58.
- **Pruning noise.** The spread of the 20 random removals is at most 0.048 on any scored state (N_1e-4_s2/1000,
  K 15), so "at most 0.05" holds. On P states the random-removal mean differs from acc0 by at most 0.00025.
- **Controls** (lines 91-102). LOW B_a reads 0.99975 and P_a 0.960; scratch takes 2,300 steps. Pruning on LOW reads
  0.9425 and 0.79575; the probe on LOW reads 0.6645 for count a (block 5) and 0.2785 for count b (block 2). The
  identity-prompt fine-tunes read S 0.937 and 0.930, with probe 0.6655. The truth set has 35 states and 0
  disagreements. P task accuracy is 1.000 for every run at 10K and at least 0.99875 at 1K. N at 1e-5, 1K reads
  0.480-0.486 and is "not yet learned" for all five seeds.
- **Update-size table** (lines 179-184). The per-cell ranges and N/P ratios 1.37-1.44, 1.15-1.25 and 1.33-1.37 match
  `.nuisance`.
- **C1 re-labelling** (lines 219-223). `c1_described.agreement` reads prune 26/26, steer 26/26 and reverse 10/12.
  The two misses are a2/N2/1000 and a2/N3/1000, both at 0.2609 (600 steps).
- **Cut-offs fixed before scoring.** `c1b_cutoffs.json` was committed at d361649, 17:34:32 UTC, and has not changed
  since. Job 46459's first scored line is at 17:37:20. The calibration logs (46399, 46443) contain no state of seeds
  2-4. `c1b.py:505-509` refuses to score unless the file is tracked, unmodified, and equal to the cut-offs recomputed
  from the calibration readings. Predictions were filed at b118997 (15:31) and amended at c6ec7d2 (15:48), before
  job 46330's first line at 15:49:30.
- **No swap-built rows.** The training rows are `c1b.task_rows`, i.e. `pcfg.task_batch(..., c_rate=0.0)` relabelled
  for P. The model-free check `existing_runs_are_the_N_recipe` confirms N1, N1s1 and M are the c_rate 0 recipe.
  Reverse fine-tuning rows are `c1.rft_rows`, natural `C a 40`. Steering rows are
  `interchange.qualifying_texts`, natural texts filtered to count(b) ≠ count(a) + 1. The report's line 246-249 is
  right.
- **Leans** (scorecard). Every lean matches `predictions_c1b.json`: C0 90/85/80, C1 85 and 40, PC1 60, PC2 45,
  PC3 60. The probe was filed as a reference only (amendment H1). The report is right that the filing's "the same
  event" was wrong: at n_c 15 the requirement is 17, which is possible, and the stop fires only at n_c ≥ 20.

## Problems

1. **Wrong: "the scored points also fall inside the ranges the calibration seeds had marked out"** (line 111, repeated
   at line 241-242: "Every scored reading lies within the range its calibration seeds marked out"). This is false
   for two instruments.
   - Pruning: the scored P maximum, 0.9203, is above the calibration P maximum of 0.8787. The scored N minimum,
     0.0576, is below the calibration N minimum of 0.0735.
   - Reverse fine-tuning: the scored N maximum, 0.6522 (N_1e-4_s4/10000), is above the calibration N maximum of
     0.6087.

   What is true is weaker, and it is all the margin argument needs: no scored reading falls inside the calibration
   gap. Every scored P reading is on or beyond the calibration P edge, and every scored N reading on or beyond the
   calibration N edge. **Fix:** replace both sentences with "No scored reading falls inside the gap the calibration
   seeds left between the classes".

2. **Misleading: the post hoc labeller is described as set "by the instruments' own rule"** (line 145; also line 245
   and scorecard line 292, "set on seeds 0-1"). `plot_c1b.py:136-158` (`cell_labeller`) does two things the filed
   rule does not.
   - It takes a gap midpoint from 2 + 2 calibration states per cell. The filed `rules.cutoff` (edge case E1) voids
     any cut-off with fewer than 3 readings in a class, so under the instruments' own rule every cell would be void.
   - In the lr 1e-5, 1K cell it labels all three scored states "existing" because only P states from that cell
     entered the truth; update size plays no part for them. So 18 of the 21 come from within-cell thresholds and 3
     from cell membership.

   The conclusion stands: in the three two-class cells, N/P is at least 1.15 on every seed. **Fix:** describe the
   rule as "the midpoint of the gap between 2 calibration states per class in each recipe cell, below the filed
   minimum of 3; the one-class cell (P at 1e-5, 1K) is labelled by its cell alone". Give 18 of 18 in the two-class
   cells beside 21 of 21.

3. **Misleading by omission: the update-size reading as possible evidence of a new capability** (lines 168-177, the
   surprise at line 259-261, and the `c1b_update_size.png` title "learning the new count moves the weights 1.15 to
   1.45 times as far"). The five N states at lr 1e-5, 1K have not learned the new count: task accuracy is 0.48, and
   they are excluded as "not yet learned". Yet their update size is already 0.0119 on every seed, against
   0.0043-0.0044 for P in the same cell. That is a ratio of 2.7, larger than in any cell where N has learned.
   `.nuisance` holds these values, but the report's table shows "not yet learned" for them and the figure leaves them
   out. The data therefore shows the size gap before the new count exists. That fits the update size tracking how
   far each task's training signal pushes from LOW (P is one relabelling away from what LOW already answers), more
   than the size of a finished new capability. It weakens the sentence "a new count may simply need a larger change
   than a constant shift of an old one". **Fix:** give the N 1e-5 1K update size in the table (0.0119, ratio
   2.7-2.8, marked not yet learned) and say that the gap is present before N is learned. Keep "it needs only the
   weights and VPD would have to beat it", which still holds.

4. **Minor: unqualified "every P state" ranges** (lines 120-122). "Every P state revives 0.60 to 0.92" and "P states
   read 0.83 to 0.96" are the scored states only. The figure on the same page shows the calibration state
   P_1e-4_s1/10000 at 0.36 for pruning and calibration P at 0.816 for steering. **Fix:** "every scored P state".

5. **Minor: the plan's design target for the recipe ceiling was missed and the report does not say so.** Plan
   section 3 set "Target: n_c ≤ 0.6 n", i.e. at most 12 of 21, so that a perfect instrument would beat the recipe by
   8 or more. The update-size labeller reached 15 (0.71 n), and the best possible margin was 6, not 8. The report
   gives 15 and the margin of 6 but not the target. **Fix:** add one clause to the truth-set paragraph (line 88-89)
   or to "How compelling".

6. **Minor: steering's noise is quoted from C1 when C1b has its own** (line 113-115). C1b's fit-versus-score
   difference at the chosen block is at most 0.021 on the 21 scored states and at most 0.020 on calibration, against
   the quoted 0.03 from C1. Using C1b's own number gives about four noise widths, not three, so the report
   understates its own margin. **Fix:** optional, quote C1b's 0.021.

Not a problem, noted for the lead: claim (b) "VPD can at best tie" is right for agreement counts on this set. The C1
states (described only, confounded, some swap-built) are one place where a cheap method is below ceiling: reverse
fine-tuning reads 10 of 12 there. That fits the report's "What next" item 2, which asks for a set where the answers
do not give the label away.

`check.py` on the report flags only sentence density (49 sentences over its number or word count, 4.7 numbers per 100
words) and no retired terms. I did not treat these flags as problems.

## Not checked

- I did not re-run any instrument or load any model. Every reading was taken from the committed JSON and logs.
- The notebook section "Stage C1b" in `notebooks/E55_pcfg_anchors.ipynb` and the `TRIED.md` and `PENDING.md` rows.
- `src/commute/revival.py`'s `cutoff`, `ceiling` and `score` beyond what their outputs show. The outputs match the
  filed rule's numbers at every point I recomputed.
