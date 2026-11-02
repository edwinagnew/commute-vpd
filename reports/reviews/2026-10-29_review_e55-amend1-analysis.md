# Review: E55 Amendment 1 analysis, independent check

2026-10-29. Reviewer did not write the code, the run or the analysis. Scope: section "Amendment 1" of
`reports/analyses/2026-10-29_analysis_e55-stage-a.md` (lines 367-653), the updated opening (line 10), answer
(lines 18-20), Stage A "What we learned" (lines 270-271), "What next" (lines 315-317), "Not run" (line 341), and the
Amendment 1 cells of `notebooks/E55_pcfg_anchors.ipynb`. Sources: `experiments/pcfg_anchors/results/amend1.json`,
`predictions_amend1.json`, `a2_W1.json`, `a2_W1s1.json`, `a2_W2.json`, `results/large/amend1/answers/*.npy`,
`counts.npy`, `cluster/logs/commute-e55-a2-2-44329.out`, `cluster/logs/commute-e55-amend1-44439.out`. Check script
(numpy only, no model): `~/commute-data/scratch/reuse-lead/e55-a1-check/check.py`.

## Verdict: supported with corrections

Every AM verdict re-scores the same from `amend1.json` by the filed rule, and nearly every number matches the data;
three are off by one in the third decimal. The unfiled one-third rule is clearly labelled exploration, is used for no
label, and its fit reproduces exactly. One claim goes further than the reads allow: the report states as a finding
that the all-spurious answers "keep depending on the count of a". Behaviour fits a rule of the counts of b and c
equally well. The Stage C part is framed as options for the user, not a decision.

## Findings, most serious first

1. **"Depends on the count of a" is stated as found, but behaviour cannot tell it from a rule of b and c** (medium).
   Lines 10, 18-19, 380, 487-488, 556, 270-271; notebook "The answer" and "What we learned from the amendment".
   Evidence: since a + b + c = 40, every rule of the counts is a function of any two of them. The one-third rule
   (2(a+1)+b)/3 is also (82 − b − 2c)/3, and the midpoint rule can be written in b and c too. The report says behaviour
   cannot tell whether the *second* count is b or c (lines 489, 573, 591). It never says the *first* need not be a. The
   count-a patch cannot separate this either: Stage A's own blind-to (line 291) notes a count-a swap at fixed b changes
   c. Reading a as one of the two counts rests on LOW computing count(a) (PA1), which is a reasonable prior but was not
   read here. Fix: say "a function of two of the three counts". Say that taking the count of a as one of them rests
   on LOW's count of a, not on these reads. Soften "never drop the count of a" (line 556) and the opening and answer
   sentences to match. The third-class name "combines the old count with a second one" (line 570) carries the same
   assumption, so the decision card should state it.

2. **Three third-decimal mismatches** (low).
   - Line 518 (trajectory table, 10K, midpoint share) reads "0.502, 0.502". The W1s1 re-run is 0.50145
     (`.trajectory.W1s1.states["10000"].rule_read.mid.S`), so it should read 0.502, 0.501. The same applies to line
     633, "0.221 → 0.502", which should read 0.221 → 0.501.
   - Line 517 (7K, midpoint share, seed 0) reads 0.486; the value is 0.48547, so 0.485.
   - Notebook, Amendment 1 scorecard, AM2 row, reads "0.502, 0.503". The saved W1 10K is 0.50145
     (`.saved.W1_10000.mid.S`), so it should read 0.501, 0.503, as the report's line 626 already has it.

3. **The re-run's divergence is put down to "nondeterministic" operations; other causes were not ruled out** (low).
   Lines 548-550, 605. The step-500 losses are confirmed (0.5977 in job 44329 log line 74 against 0.5955 in job 44439
   log line 21). The data stream is deterministic per step (`Batches` seeds each chunk by `[seed, chunk]`), so the
   runs part on identical data before any resume. That fits nondeterministic kernels, but also a different GPU or
   different compiled kernels, each deterministic on its own card. After 1K the runs also resume at different steps
   (Stage A once at 1K; the re-run at 1K, 2K, 3K, 5K and 7K). Fix: "the runs part by step 500 on the same data; whether
   from nondeterministic kernels, a different card or different compiled kernels was not measured".
   The learning-rate sentence for W2 at 1K (lines 546-547) is fine. `a2_W2.json` arm is lr 1e-4 against 1e-5, with the
   same prior, c_tr and seed, and the sentence says the cause was not tested.

4. **The "correlation 0.79 at fixed c" is computed correctly, but its method is unstated and it barely separates
   rules** (low). Line 490; writer's `cfixed.py`. Recomputed: the unweighted mean of the per-c correlations, over the 7
   c values with more than 30 rows and more than one answer, is 0.794 (W1s1 0.795). The row-weighted mean is 0.777,
   and the pooled within-c correlation is 0.780. It supports "not a function of c alone", since such a rule has no
   variation at fixed c. But the one-third rule itself scores 0.78 and the midpoint rule 0.82, so the number says
   nothing about which two-count rule. A clearer number for the same claim: the best possible rule of c alone (the
   per-c mode, fitted) matches W1 10K on 0.765 of rows (W1s1 0.768), against 0.978 for the one-third rule. Fix:
   name the statistic, or replace it with that comparison.

5. **The fast arm at 1K "sits between the slow arm's at 3K and at 5K"** (low). Line 546. W2 1K reads S_a+1 0.155
   (slow 3K: 0.217, 0.148; 5K: 0.039, 0.076), S_mid 0.453 (slow 5K: 0.440, 0.440, so above both), lower half 0.000
   (slow 3K: 0, 0). Roughly at the slow arm's 3K-5K, not strictly between them on every share. Fix: "close to the slow
   arm's 3K to 5K readings".

6. **Two readings bear on the third-class decision but go unmentioned** (low, input to the user's decision, not an
   error). (a) The fast all-spurious state at 1K matches the one-third rule on 0.899 of all rows and on 90 of the 92
   rows where that rule differs from the four filed ones. The fast arm then moves on to the midpoint by 10K. So the
   one-third rule may be a stop the fast arm passes through, not a separate end state of the slow arm. (b) The slow
   state is still moving at 10K. Its lower half, rounded up, goes 0.113, 0.198, 0.356 at 5K, 7K and 10K, and its
   one-third share goes 0.962, 0.973, 0.978 (seed 0). "Coherent function" (line 558) describes the state at 10K, not
   a settled one. Suggest one line in "What it means for Stage C" or "How compelling".

## Checks that passed

- **(1) AM verdicts.** Re-scored from `amend1.json` against `predictions_amend1.json`. AM0 formula: 8 of 8. AM0
  synthetic: 6 of 6, and the log shows the c_floor and c_ceil answerers read "fixed rounding". N1 10K: overall and S_b
  1.0. LOW: overall_a 0.999875 on both prefixes. All of AM0 is met. W0 is as expected (B_a 0, B_b 0.243, P_b 0,
  P_a 0.959, labels neither / not through). AM1 is met: S_a+1 is 0.8215 and 0.8622 against 0.7285, and no other rule
  follows. AM2 is missed: S_mid 0.5015 and 0.5029, lower ceil 0.350 and 0.367. AM3 is met at 10K (0.998, upper 0.984,
  lower 1.000, route mid) and missed at 1K (0.155). AM4 is met: saved states 5 mixed and 1 towards a+1; trajectory 12
  of 12 mixed. AM5: reproduces met (8 of 8 per seed; the largest gap is 0.002, P_b at 10K); same_run missed (4.49e-4,
  2.30e-3; 4.13e-4, 2.44e-3); leaves_late met; rule_switch missed. The B_b gap of 0.243 against 0.269 at "about two
  standard deviations" holds, since the null was read on different texts (seed `[EVAL_SEED, 31, i]` against
  `[EVAL_SEED, 51]`). The Stage C counts are right: 5 neither = M 1K, M2 1K, W1 10K, W2 1K, W2 10K, of which three
  are all-spurious.
- **(2) Numbers.** The full-readings table (lines 497-506) matches to three decimals on every cell. The trajectory
  table (lines 511-518) matches apart from finding 2. The exploration offset table (lines 526-537) matches every
  count. Seed 1 differs by at most 21 rows in any cell, as stated (b − a = −1: 54 against 33). "All but 54 of 3,000"
  is confirmed, and so is the analytic claim that the midpoint and one-third rules give identical B_a and B_b (both
  equal a + 1 exactly for d in {−1, 0, 1} and b only for d = 0).
- **(3) One-third rule.** It reproduces: 0.9779 (W1 10K) and 0.9809 (W1s1); 92 distinct rows (b − a in −6, −5, 5,
  7), answered 92 of 92 by both seeds; W2 10K 0.848; along the re-run 0.572 and 0.554 at 1K, 0.962 and 0.941 at 5K.
  `amend1.py` contains no such rule, and no label, verdict or scorecard row uses it. It is labelled post hoc at lines
  383-384, 470, 480, 491-492, 557, 571 and 596-597, and option 2 makes the slow state's membership wait on filing and
  fresh rows. The rearrangement (a + 42 − c)/3 is correct.
- **(5) Stage C.** Framed as three options for the user (lines 574-580, 609), and the class is "candidate" in the
  opening. No label changes.
- **(6) /writeup.** `check.py` gives style flags only (sentence length, numbers per sentence) and no retired terms.
  The opening paragraph is now four sentences, against a three-sentence rule; that is minor and left to the writer. A
  "Not run" list is present.
