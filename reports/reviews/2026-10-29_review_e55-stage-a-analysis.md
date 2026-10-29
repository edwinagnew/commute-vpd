# Review: E55 Stage A analysis (independent check)

Reviewed: `reports/analyses/2026-10-29_analysis_e55-stage-a.md` (cited as `report:L`), `notebooks/E55_pcfg_anchors.ipynb`,
against `experiments/pcfg_anchors/results/*.json`, `predictions.json` (four filings), the plan
`reports/plans/2026-10-29_plan_reuse-stage1.md`, `experiments/pcfg_anchors/run.py`, `src/commute/interchange.py`,
`src/commute/probe.py`, `src/commute/pcfg.py`, `experiments/pcfg_anchors/plot.py`. Reviewer did not write any of it.

## Verdict: supported with corrections

All the verdicts in the scorecard are correct when re-scored from the JSONs by the filed rules. Every number in the
report matches a JSON except one range. The filing order holds: predictions were filed at 19:32/19:34, 20:06 and
20:27 (A2 bars only), before `a1_*.json` (19:48), `a1c_*.json` (20:11/20:25) and `a2_*.json` (20:31 on).

Three statements need correcting before the report is used to plan Stage C. (1) The headline says fine-tunes that
learned to count b never used the count of a. The report's own table contradicts this: M3 at 1K reads "through".
(2) The "count of c" reading proposed for the all-spurious states at 10K is already ruled out for W1 by numbers in the
JSONs. A reading that fits them uses the count of a, so "it has found something else" is not supported. (3) PA4
passes by its filed rule, but the count-b probe margin lacks the control that the count-a margin has. That makes
"reuse a readout LOW already had" stronger than the data allow.

## Re-scoring (all from the JSONs, filed rule, strict inequalities)

- **Bars.** These were recomputed from `a1c_patch.json`. B_b: n₀ = mean(0.268, 0.284, 0.255) = 0.269, sample sd
  0.01453, lo = 0.269 + max(0.0436, 0.0731) = 0.3421, hi = 0.6345. The other three bars match too. l* = 5 for both
  (blocks 1-5 only; block 5 is the maximum). PA1 patch means by block match `report:L121-124`.
- **Labels.** All 44 labels (22 states × 2 counts) were recomputed from B and P against the filed bars. All match
  `report:L172-193` and the JSONs' `label` fields. Every P rests on 1,000 pairs (none void).
- **PA1.** Pass. All ten `PA1_rows` and three probe rows are true. m₊ = 0.520 ≥ 6 × 0.0088 = 0.053.
- **PA2.** Pass, 7/9. M3 is off by 5.95, W1 by 17.675, W2 by 24.05. The other cells are within 1 point.
- **PA3.** Miss, 4/7 (W1 1K through; W1 10K, W2 1K, W2 10K neither; N1 both not through; R 10K through count b). No
  reversal: no W state reads "not through" count a, and N1 never reads "through".
- **PA3 null.** Did not occur. No state, at either count, has one check above hi and the other below lo.
- **PA4.** Pass by the rule. At block 2, m = 0.2765 ≥ 3 s = 0.0325 and ≥ 0.5 m₊ = 0.2599. s is at the binomial
  floor (0.0108). Block 3 (0.2657) also clears.
- **Stage C.** The 16 LOW seed-0 states split 2 through (M3 1K, W1 1K), 9 not through and 5 neither. The plan
  expected 6 / 8 / 2 (plan L176). With n = 11, P(Bin ≥ 9) = 67/2048 = 0.033 and P(Bin ≥ 8) = 0.113, so a = 9. The
  majority raise gives 10. This is correct.

## Findings (most serious first)

1. **The headline is contradicted by the report's own table** (`report:L8`, `L14`, `L155-157`, `L233`; notebook "The
   answer" and "What we learned"; `figures/labels.png` title). The report says "Fine-tunes that learned to count b never
   used the old count" and "Every fine-tune that learned to count b does not go through the count of a". M3 reaches
   99.85% on count b at 10K. Yet at 1K it reads **through** count a: B 0.631 > 0.500 and P 0.594 > 0.483
   (`a2_M3.json` states.1000.check_count_a). The report itself counts it as one of the two "existing" states
   (`L207`). M2 at 1K has P_a 0.519 over its upper bar. `L157` calls the slow learners' 1K answers "still partly
   count(a) + 1", which understates a "through" label. The plan's own leans expected exactly this (plan L138: "M2, M3
   at 1K: through count-a"). **Fix:** say that fast fine-tunes never went through the count of a. The slow fine-tune
   with 80% spurious rows went through it at 1K and had left it by 10K, just as the all-spurious slow fine-tune did.
   This also changes `L233-236`: going through the old count early and leaving it is the pattern of every slow arm
   with spurious rows, not only the all-spurious one.

2. **The count-c reading of W1 at 10K is already ruled out, and the data point the other way** (`report:L163-168`,
   `L236`, What next 2 `L270-272`). Any answer that is a function of the count of c alone gives B_b of at least
   about 0.15 when it reaches B_a ≈ 0.40. Checked as follows. `pcfg.py` has only three terminals, so a + b + c = 40
   on *every* row, natural rows included. On qualifying natural inputs:
   - the count-c answer (41 − c)/2 with fixed rounding gives (B_a, B_b) = (0.261, 0.149) rounding up, or (0.149, 0.261)
     rounding down;
   - the best possible function of c alone with B_b ≤ 0.01 reaches B_a 0.105, and with B_b ≤ 0.05 it reaches 0.260
     (exact optimisation over 65K sampled qualifying windows, no model;
     `~/commute-data/scratch/reuse-lead/e55-check/conly3.py`).

   W1 at 10K reads B_a 0.399 and B_b 0.001, and its seed-1 run reads 0.399 and 0.000. That is far outside what any
   count-c-only answer can give. One rule fits both numbers: the midpoint of count(a) + 1 and count(b), rounded towards
   count(a) + 1, gives (0.408, 0.000) on the same windows (`conly.py`). It is also consistent with W1's 10K accuracy
   (0.17675, right on the non-qualifying rows and almost never elsewhere) and with the block-6 patch depending on both
   counts (0.64, 0.57). That rule reads count(a) on every input through a function other than +1. The behaviour check
   is blind to such a function (`L245-246`), and the patch at block 5 is only partly sensitive to it. So `L236`, "by
   the end it has found something else that fits the same rows", is an untested cause. The data are as consistent with
   W1 at 10K still using the count of a. W2 at 10K (0.292, 0.102) is not excluded by this bound.
   **Fix:** drop the count-c reading for W1, or say it is excluded. Replace `L236` with "the checks cannot say whether
   it still uses the count of a". Redirect What next 2 to the per-row answer table of W1 and W2 at 10K against
   (count(a), count(b)), on the saved states. That one reading separates the midpoint rule from the alternatives. If
   W1 at 10K uses count(a) through a midpoint, the "existing labels are scarce" conclusion (`L236-237`) weakens.

3. **The PA4 interpretation leans on asymmetric controls** (`report:L214-219`, `L234`, `L251-253`; notebook
   possession section; `figures/probe.png`). The count-a margin subtracts NEVER's count-a probe. NEVER is a model with
   no count-a task, yet it still decodes count a at 0.47-0.52 (`a1c_probe.json` raw.count_a NEVER.acc, blocks 2-6)
   against about 0.34 for shuffled labels. The count-b margin subtracts the shuffled-label probe only. No analogue of
   "a model that never had the task" is subtracted (none exists in this design), and the notebook's possession
   paragraph omits the NEVER control entirely. LOW's count-b probe peaks at 0.626. That is about 0.11 above the 0.516
   NEVER reaches for a count it was never asked for. The cross-model comparison is not the filed rule, but against
   that kind of control the gap is under half the 0.26 bar. Separately, count(b) can be guessed from count(a) alone at
   0.40 on natural windows (Bayes rate, `cond.py`; majority 0.35; corr(a, b) = −0.49). So part of the margin over
   shuffled labels can come from LOW's count of a. PA4's "pass" stands by its rule.
   **Fix:** state the asymmetry in "What the probe is blind to". Replace "LOW holds a linear readout of the count of b"
   and "reuse a readout LOW already had" (`L217-218`, `L234`) with "count b decodes from LOW's residual above shuffled
   labels, by less than a no-task model decodes count a". In `probe.png`, say in the legend that the two curves
   subtract different controls; the "no better than shuffled labels" label at 0 is wrong for the count-a curve.

4. **"Mostly on natural rows" is wrong** (`report:L134`; notebook Table 6 reading). N3 and M3 train on 80% spurious
   rows. **Fix:** "every fine-tune with any natural rows".

5. **`report:L136` juxtaposes W1's 10K accuracy with "the score of always answering count(a) + 1"**, which implies
   that W1 at 10K does so. Its B_a at 10K is 0.399, so it does not. The match (0.17675 vs 0.1765) arises because it
   answers correctly on the non-qualifying rows (finding 2). **Fix:** "equal to that score, although by 10K it gives
   count(a) + 1 on only 0.40 of qualifying inputs".

6. **Wrong range** (`report:L197`): "0.36 to 0.43 of pairs" at 10K. The values are W1 0.429, W1 seed 1 0.427 and W2
   0.377 (0.358 is W2 at 1K). **Fix:** "0.38 to 0.43".

7. **The spurious construction is misdescribed** (`report:L39`; notebook). `pcfg.spurious_texts` keeps the natural
   count(b) as the label and sets count(a) to count(b) − 1 by a↔c swaps in the window (`pcfg.py:164-198`). count(b)
   is not set. The sum to 40 holds on every row, natural rows included (`L166` and `L249-250` present it as a
   spurious-row property). This matters for finding 2. No check on these inputs can vary the count of c with the
   counts of a and b both fixed, so "through the count of c" versus "through a and b" can only be told apart by a
   mechanism-level test, not by B or P. **Fix:** describe the construction as coded. Add to the patch's blind-to
   that this also holds for the count-b patch (at a fixed count of a, changing b changes c), so R's and N1's "through
   count b" carry the same ambiguity.

8. **Causes stated as fact.**
   - `L114-115`: "Our window counts sit between 9 and 17, so the two counts agree more often than in the paper". The
     paper's count spread was not measured here. Say "probably because".
   - `L113`: "asked for b, it answers its count of a" is well supported but rests on an equality of rates. The report
     can add a second, independent check: the B_b null 0.269 equals 0.2215 / (1 − 0.1765) = 0.2690, the share an
     exact count(a) answer would score on qualifying inputs.
   - `L200-201`: "the count of b reaches the answer later than block 5 on more inputs" is one reading of P_b(5)
     falling while P_b(6) stays 1. It is untested; say "consistent with".

9. **Noise of the A2 readings is unquantified** (`report:L241-244`). Each A2 state is read once, on one set of 1,000
   pairs and 1,000 inputs shared by every state, so "few labels are decided by noise" has no number behind it. The
   binomial sd is about 0.016 near 0.5 and about 0.009 near 0.1. W2 1K B_a (0.012 under hi, ~0.75 sd) and W1 1K P_b
   (0.004 over lo, ~0.4 sd) are within noise. Neither decides a count-a label: W2 1K's P_a is 0.069 under its bar.
   M 1K B_a (0.118 vs lo 0.100, ~2 sd) is the next closest. **Fix:** give the binomial sd and say that the near-bar
   cells are within it, and that the Stage C labels for count a do not depend on them.

10. **"Not run" and "blind to" are incomplete** (`report:L245-255`, `L294-302`). Missing items:
    - B_a's positive reference is LOW's plain count-a accuracy on `C a 40` (`run.py:326`). That is a different
      quantity from the A2 reading, which is the share equal to count(a) + 1 on qualifying `C b 40` inputs. No
      positive control of the A2 reading itself was run; W1 at 1K (0.879) shows it can read high.
    - The P_b null is LOW on `C a 40` (`run.py:257`), not on the arms' own prefix `C b 40`. A count-b patch null on
      `C b 40` was not run.
    - Per-state resampling of B and P (finding 9).
    - A patch at the text positions, or at any position other than `=`.
    - The asymmetric probe controls (finding 3) and the a + b + c = 40 ambiguity for both patches (finding 7).

## Code against the report

The code computes what the report says it measures. P has all pairs built as its denominator and counts a pair when
the patched output equals the donor's output and that differs from the clean output (`interchange.py:136`). B is
computed on the first 1,000 qualifying `C b 40` inputs, with g = count(a) + 1 and count(b) (`interchange.py:29`,
`122`). Patch candidates are block outputs at `=` only (`interchange.py:84-87`, `109`). l* is taken over blocks 1-5
(`interchange.py:158`). The bar rule matches the filed E13/E14 (`interchange.py:168-175`). Labels use strict
inequalities (`interchange.py:178-187`). The probe margin is acc − max(controls) per seed, with s floored at the
binomial sd at 2,000 (`probe.py:94-99`). `plot.py` reads only the JSONs and the filed bars. No mismatch was found.

## /writeup

`check.py` flags 30 sentences, mostly length; none is serious. The only serious write-up problem is finding 1: the
80-word answer and the plain-words paragraph state a conclusion that the report's own table contradicts.
