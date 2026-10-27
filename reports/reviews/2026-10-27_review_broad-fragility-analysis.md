# Review: E39 analysis (broad fragility: random moves, reverts, update shape)

*2026-10-27, independent check by a fresh agent that did not write the code, run or analysis. Read-only on everything
except this file. Scope: `reports/analyses/2026-10-27_analysis_broad-fragility.md` and the markdown of
`notebooks/E39_broad_fragility.ipynb`, against `experiments/broad_fragility/results/predictions.json`,
`results/broad_fragility.json`, `results/large/run.log`, `run.py`, `src/commute/fragility.py`, and the code paths
they call (`row_copy.read_state`, `component_training.train` and `components_with`, `faithfulness.model_state`,
`vpd.mean_ci`). E37 reference numbers checked against `experiments/teacher_spread/results/teacher_spread.json` and the
E37 Amendment 1 analysis.*

**Verdict: the numbers and the scored verdicts are right, and the code behind the main claims is sound; the
headline reads more into the random move and the LoRA arm than they carry, and two sentences contradict the table.**
Every number in the report's tables matches the JSON or the log. P1 to P4 are scored exactly as filed. One scoring
call should change: P6 as a whole is a miss, not unscored. The main over-claims are "not general brittleness" from a
random move that does a twenty-fifth of the damage, and "a low-rank update of the same behaviour" for a LoRA that differs
in gain, in plain-text change, and in how far teacher training moved the subcomponents.

## Verdicts recomputed

Recomputed with `jq` from `results/broad_fragility.json` by the rules in `predictions.json`.

| id | filed rule | recomputed read | verdict | analysis says |
|---|---|---|---|---|
| stage 0 (i) | spread ≤ 0.10, blocks within 1% of 0.6794, 0.7182 | spread 0.07482; blocks 0.67939, 0.71825 | pass | pass, agrees |
| stage 0 (ii) | gain within 0.005 of 0.1301; check within 3% of 2.2011; s within 0.01 of 0.998 | 0.12995; 2.20110; 0.99779 | pass | pass, agrees |
| stage 0 (iii) | \|check − 1\| ≤ spread; \|s\| ≤ 0.01 | 1.0 exactly; 0.0 | pass | pass, agrees |
| P1 | mean of x1 moves ≥ 1.61 | (1.02886 + 1.06885)/2 = 1.04885 | miss | agrees |
| P2 | x0.5 below the x1 mean | 0.98817 < 1.04885 | holds | agrees, "within the noise" is right |
| P3 | no layer share > 0.6 | 0.7047, 0.3781, 0.2530, 0.1076 | miss | agrees |
| P4 | rare − random ≥ 0.2 | 0.62937 − 0.97339 = −0.34402 | miss | agrees |
| P5 | SGD and LoRA stable rank each < Adam's 28.517 | LoRA 1.933; SGD none | LoRA half holds; whole unscored | agrees |
| P6 | SGD and LoRA teacher-trained check each ≥ 1.61 | LoRA 1.2726; SGD none | **whole: miss** (one conjunct already false) | "not scored": should change |
| P7 | SGD and LoRA do-nothing each ≥ 1.61 | LoRA 2.6531; SGD none | LoRA half holds; whole unscored | agrees |

P6's rule is a conjunction over the two arms ("for SGD and for LoRA, each"). The LoRA half reads 1.27, so no SGD
result can make P6 hold. The filed rate rule ("none: that arm's P5-P7 are not scored") leaves the SGD halves unscored,
not the whole of a prediction already falsified by the other arm. `run.py:418-420` sets a whole prediction only when
both halves exist. That is a coding choice, not the filed rule. P5 and P7 are rightly unscored as wholes, since their
LoRA halves hold.

Predictions provenance: `predictions.json` was committed at 15:30:59 (`60143f5`) and refiled at 15:32:25 (`3cf7ad4`).
The refiling changed only stage 0 (ii)'s wording, before any run, and the commit message says so. The run log starts
15:44:47. "Unedited" since the run holds. sha256 prefixes in the Provenance section all match the files on disk.

## Numbers checked

Every cell of the analysis's four tables, every number in the prose, and the scorecard were checked against the
JSON, the log, and E37's `teacher_spread.json` (do-nothing 4.868, stochastic 1.106, damage 0.235, size 0.049; scaled
adjoint split 1.054, s 0.804, gap 0.111, damage 0.050, stochastic 1.048, size 0.0399). E37's Amendment 1 masked
control (3.14 at damage 0.007) matches its analysis. Revert counts add up: layers 9,881, attention plus MLP 9,881,
rare plus common 9,881. One arithmetic slip:

- **Analysis line 178:** "The layer shares sum to 1.444." The sum is 1.4434, so it should say **1.443**. Line 148's
  "1.44" is right.

## Code paths behind the main claims

No bug found that would invalidate a claim.

- **Random-move scaling** (`fragility.py:21-55`). `_scale` solves ‖aA + a²B‖_F = target exactly from the quadratic
  expansion by bisection. The JSON's `norm_miss` is at most 7.7e-8. Each move's `size` equals the trained state's,
  0.03648. The x0.5 move reuses seed 0's noise, so it lies along the same direction up to the second-order term.
- **Subset revert** (`fragility.py:58-65`). `torch.where(m, U, U2)` with m of shape (n,) broadcasts over the last
  axis of both U (d_out, n) and V (d_in, n), so whole subcomponents are reverted. Column order is the same in
  `alive_uv`, `components_with` and `g[k][alive[k]]`, since all three use the same boolean mask. Revert-all reads
  every harness ratio as exactly 1.0, which confirms the path.
- **Importance halves** (`run.py:273-287`, `fragility.py:68-80`). `vpd.mean_ci` runs outside any `model_state`, after
  stages that restore the model on exit (`faithfulness.py:35-49`). `component_training.train` never writes target
  weights and removes its hooks. So Γ is read at the unchanged model, as stated. The rare and random halves are
  matched in count (4,935 each), not in how much weight change they carry. That is the confound the analysis names
  under Surprises.
- **Share of excess** (`run.py:405`). This is the filed formula, and `plot.py:181-185` re-derives it to 1e-12.
- **Size** (`run.py:180`). This is the largest relative change of *any* matrix, so after a revert the largest matrix
  can be a different one. This affects one passage (below).

## Errors and over-claims, with the fix

1. **"Any half" contradicts the table.** Analysis lines 28-29: "Putting back any half of each matrix's subcomponents
   removed nearly all of it." Lines 148-149 say the same. The rarely important half removed 0.63, which is not
   "nearly all". The notebook's "The answer" and "Which subcomponents" cells repeat it. **Fix:** "Putting back a
   random half, or the often important half, removed nearly all of it, 0.97 and 0.96. The rarely important half
   removed 0.63." Also note that the halves were taken over all 24 matrices. "Most of the MLP changes together" is
   inferred from all-matrix halves, and MLP-only halves were not run (the Not run list already says so).

2. **The surprise box puts the MLP result under a filed prediction it does not test.** Lines 37-39 list "The MLP
   changes alone carry the excess, and the attention changes hide part of it" as a filed prediction that went the
   other way, against "we expected the excess spread out". The filed prediction is P3, about *layers*. Its miss is
   layer 0 removing 0.70 against a 0.6 cap, which is inside the 0.2 share noise, as line 259 says. The attention and
   MLP split had no filed prediction. An MLP-only excess spread over all four layers does not contradict "spread over
   layers". The notebook box says the same. **Fix:** let that bullet say P3 missed narrowly, within the noise, and
   move the MLP and attention result to Surprises as an unpredicted observation. "Four filed predictions went the
   other way" then becomes three clear misses (P1, P4, the LoRA half of P6) plus one narrow miss (P3).

3. **P6 scoring.** Line 207: "By the filed rule the SGD arm's three predictions are not scored, and so neither are
   those predictions as a whole." Scorecard line 305 has "LoRA **miss**; P6 not scored". The notebook scorecard has
   the same. **Fix:** "P6 misses as a whole: its rule needs both arms at or above 1.61, and the LoRA reads 1.27. P5
   and P7 are not scored as wholes; their LoRA halves hold."

4. **"Not general brittleness" is stated flat in the answer.** Line 27 says "The failure is not general brittleness
   of the decomposition." Lines 237-238 say "this points away from 'the decomposition is too brittle'." The random
   moves are isotropic per side. They change plain-text predictions a twenty-fifth as much (damage 0.009 and 0.010
   against 0.234). So a pass is what both hypotheses predict, and the test has little power to separate them. The
   analysis gives this caveat in section 1, but not in the answer, and only partly in "What we learned". The E37
   masked control, which the report cites, cuts the other way. A trained move of half this size (largest change
   0.018, damage 0.007) failed at 3.14. So the decomposition *is* fragile along some small, low-damage directions.
   **Fix:** in the answer, "not brittleness to any move of this weight size: an isotropic random move of matched
   size per matrix passed, but it changed the outputs a twenty-fifth as much". In "What we learned", say the data fit
   both "the update is the problem" and "the decomposition is fragile along the directions training moves it in,
   though not along random ones". Only the damage-matched random move (What next, item 2) separates the two.

5. **"A low-rank one of the same behaviour is not" (lines 238-239).** The LoRA is not the same behaviour. Its gain is
   14% higher, its plain-text damage is half (0.121 against 0.235), and teacher training moved its subcomponents less
   than half as far (size 0.017 against 0.036). **Fix:** "a rank-8 LoRA of the same task, which changes plain text
   half as much, is not". The notebook's "What we learned" has the same sentence.

6. **The gain-mismatch argument is an untested assumption** (lines 196-198). "A larger gain to imitate should make
   passing harder, not easier, so this mismatch does not explain the pass." Nothing here shows that the check grows
   with lowercase gain. The LoRA's larger gain came with less plain-text change. **Fix:** "If a larger gain makes
   passing harder, this mismatch does not explain the pass; that was not tested." Also drop the doubled "So … So"
   (lines 197-198).

7. **A third confound for the LoRA pass, the size of the teacher-trained move, is not named as an alternative.** Line
   230 reports it (0.017, under half the Adam one's) but does not tie it to the pass. Every Adam-derived state in
   test 2 whose largest remaining change is at most 0.022 passed (random half 0.022, common half 0.019). **Fix:** add
   it to the caveats beside damage in section 3 and in Surprises. Evidence in hand also bears on the damage confound,
   and the author may add it, labelled as a cross-state comparison and not a test. At damage 0.10 to 0.13, the Adam
   reverts read 1.45 (rare half), 2.07 (layer 3) and 2.70 (attention). The LoRA's trained decomposition reads 1.27 at
   0.124. So plain-text damage alone does not set the check here.

8. **Damage is not named as an alternative for the MLP result** (lines 145-150, Surprises line 268). In test 2,
   every state with damage at most 0.054 passed: MLP revert 0.025, common half 0.021, random half 0.054. Every state
   with damage of 0.098 or more read at least 1.35. Reverting the MLP matrices cuts damage from 0.234 to 0.025. So "the
   MLP subcomponents carry it" and "the check falls with how much plain-text change remains" both fit. The spread at
   similar damage (item 7) shows damage is not the whole story. **Fix:** name it in Surprises beside the compensation
   cause.

9. **"P4 failed, so it is not the frozen Γ's rarely important subcomponents"** (lines 309-310). P4 tested one ranking,
   Γ's mean importance at the unchanged model. Under it, all three halves average low importance (0.003, 0.015,
   0.028), and the halves carry unequal weight change. The analysis says both things under Surprises. **Fix:** "P4
   failed: ranking by importance at the unchanged model gives claim B no support; Γ at the trained state was not
   tested."

10. **Size passage** (lines 180-183). "So the rarely important half carries less of the largest matrix's change." Size
    is the largest remaining change of any matrix, and the matrix that is largest can differ between reverts. **Fix:**
    "carries less of the largest remaining per-matrix change". The hedge that follows can stay.

11. **SGD wording.** Line 31 has "because none of its rates learned", and the notebook's SGD cell has "By the filed
    rule, its arm was not run". The arm ran its three-rate grid. All three rates were too large: even 0.3 moved the
    weights 3.5 times as far as Adam's (0.17 against 0.049) and lowered held-out log-probability. **Fix:** "every rate
    in its grid was too large and lowered held-out log-probability, so the rest of the arm was skipped and its
    predictions are not scored". The notebook sentence is wrong as written.

Minor points, optional. Line 98's "bit for bit" is inferred from exactly equal ratios, not from comparing weights;
"exactly" is enough. Lines 155-157's "Both halves are off at most positions" is inferred from mean importance; a low
mean also fits small values everywhere. A point the noise section could add: every state is read at adversary seed 0,
and W0's seed 0 was the lowest of its three seeds (ratios 1, 1.075, 1.011). So the ratios are biased up, not down, by
up to 7.5%, which makes the passes conservative.

## What holds

The scored verdicts for P1 to P4, the stage 0 controls, and every tabled number. The finding that reverting the
attention matrices raises the check, from 2.20 to 2.70. That rise is larger than both E37's 14% batch-order spread and
W0's seed spread. The finding that the rarely important half, ranked at W0, removes less of the excess than a random
half. The finding that the attention-only state (MLP reverted) passes at 1.00 while keeping 0.47 of the behaviour.
The LoRA trained decomposition's pass at 1.27. That is a single read, but it sits well below the Adam one's
batch-order range of 1.88 to 2.20, and below the Adam one relative to its own do-nothing (excess 0.16 of do-nothing's,
against 0.31). The caveats already in the report are accurate: one adversary from one start, matching by weight norm
only, reverts that are not additive, one LoRA seed and rate.

## Bottom line

The run and the arithmetic are clean. No code path behind the random move, the reverts, the importance halves or the
share of excess has a fault, and P1 to P4 are scored exactly as filed. P6 should be scored a miss as a whole, not
"not scored". The write-up's direction of reading is defensible, but its answer and "What we learned" say more than
the tests can. The random move passed while doing a twenty-fifth of the damage, so it cannot by itself rule out a
brittle decomposition. E37's masked control shows the decomposition fails along some small, low-damage directions.
The LoRA's pass is confounded three ways: gain, plain-text damage, and a teacher-trained move half the size. Two
sentences claim "any half" removes nearly all of the excess when the rarely important half removed 0.63. The box puts
the MLP result under P3, which only tested layers. With the eleven fixes above, the report would claim what the data
carry: the excess needs most of teacher training's MLP changes and is not reproduced by isotropic noise of the same
weight size, and a LoRA fine-tune of the same task gives a passing trained decomposition for reasons not yet
separated.
