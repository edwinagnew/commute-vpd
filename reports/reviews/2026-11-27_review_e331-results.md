# Check of E331's results, the scaled adjoint split on the built edits (review, 2026-11-27)

I re-read E331's saved result lines and checked regating-lead's numbers and proposed paper line against them. Every
number in their message matches the files. The scaled split puts the seven edits in the same order as least squares,
at lower values. The proposed sentence is right about the scaled split. Its last clause, about the split without its
scale, is wrong and needs rewording. One filed rival is also met as written, which the page should say.

Sources: `experiments/adjoint_rho/results/items_run2_s{0..3}.jsonl` (E331) and `items_s{0..3}.jsonl` (E329), at main
41b087a3.

## Numbers checked

| Item | Least squares (E268) | Unscaled (E329) | Scaled (E331) | Scaled, over E268's needed set | c (min / median / max) | R median |
|---|---|---|---|---|---|---|
| x2 | 0.761 | 0.207 | 0.713 | 0.713 | 0.125 (one matrix) | 0.97 |
| x10 | 0.761 | 0.207 | 0.713 | 0.713 | 0.125 (one matrix) | 0.97 |
| rebuilt | 0.623 | 0.012 | 0.453 | 0.453 | 0.019 / 0.108 / 0.190 | 0.51 |
| new_few | 0.0127 | 0.0127 | 0.0031 | 0.0050 | 0.122 / 0.133 / 0.143 | 0.28 |
| new_thin | 0.0188 | 0.0133 | 0.0134 | 0.0134 | 0.352 (one matrix) | 0.44 |
| random_size | 0.0110 | 0.0018 | 0.0015 | 0.0015 | 0.022 / 0.192 / 1.039 | 0.41 |
| random_kl | 0.0106 | 0.0016 | 0.0014 | 0.0027 | 0.022 / 0.192 / 1.002 | 0.42 |

Stage 0 passed on all four pods: the do-nothing reads exactly 1 with no changed variable, and the identity interchange
gives 0 argmax changes. The pieces plus the remainder equal dW to within 3e-17, well under the filed bar of 1e-12. Every
c is positive. With the needed set held at E268's, the gain and rebuilt edits read the same as with their own set, so
their drop comes from the split, not from a different set.

## The proposed paper line

Proposed: "With the adjoint split (scaled per matrix, as defined), the gain edits read 0.71, rebuilt 0.45 and the new
edits 0.003-0.013, the same ordering as least squares at lower values; without its scale the split does not separate
them."

1. **MUST: the last clause is wrong.** Without its scale, the split still puts the gain edits (0.207) well above the
   new edits (0.013) and the randoms (0.002). What it loses is the rebuilt edit. Rebuilt reads 0.012 there, level with
   the new edits. Suggested: "without its scale, the rebuilt edit reads as low as the new edits (0.012 against
   0.013)."
2. **SHOULD: give the reach.** The scaled split reproduces 97% of the gain edits' weighted energy but only about half of
   the rebuilt edit's (R 0.51). Part of rebuilt's drop from 0.62 to 0.45 may therefore be the split's limit rather
   than the edit. A clause such as "the scaled split reaches about half of the rebuilt edit's change" keeps a reader
   from reading 0.45 as the edit's own value.
3. No verdict wording. The run is descriptive, and the line gives readings only, as P1.1 requires. Keep it that way.

## The filed rivals

- Rival a, as filed: the gain edits "above 0.761 by more than 0.003". Not met: they read 0.713.
- Rival b, as filed in `predictions_run2.json`: the gain edits "at or below their least-squares 0.761, as E329's
  unscaled 0.207 or between". Met as written, because 0.713 lies in that band. The card's version of rival b ("reads
  like the unscaled one ... within 0.05") is not met, because 0.713 is 0.51 above 0.207. The page should report both
  as read. It should not say "the scale does not rescue it": the scale brings back most of the gain edits' share and
  the rebuilt edit's place in the order.

## Not run

- The per-item check of W0 on the model the input statistics are read on (SHOULD 1 of the pre-run check). It was not
  added mid-run, so the result page lists it as not run.
- No independent recompute. This is not a headline answer-paragraph number.

## Re-check of the final page (2026-11-27)

Page `reports/analyses/2026-11-27_analysis_p167-adjoint-rho.md` at b9be934e. Every earlier item is closed. One new
MUST: a wrong ratio in the opening paragraph.

Earlier items:

- E331 MUST 1 (the unscaled split does not lose the gain edits, it loses rebuilt): closed. :158 "Without its scale, the
  rebuilt edit read as low as the new edits: 0.012 against 0.013", and :71, :181.
- E331 SHOULD 2 (give the reach): closed, :126-129, :161-164, and in the summary :6 and :18.
- E331 rivals reported both ways, no "the scale does not rescue it": closed, :198-199 and scorecard :218-220.
- E331 item 3 (readings only, no verdict): closed, :139, :181-182.
- E329 MUST 1 (run 1 named the unscaled split; the Terms' split is the scaled one): closed, :31-34, :98-100, :224-226.
- Code review SHOULD 1 (per-item W0 check): not added; stated as not run, :193-194, :203, :231-232. Closed as reported.
- Code review SHOULD 2 (quote the filed wording of rival a): closed, :135-137, :218.
- Code review "record min c and R": closed, :166-167 (c 0.125; rebuilt 0.019 to 0.19), :189 (every c positive).

New items:

- **MUST: :4 "Without its scale the rescaled edits fell to a fifth".** The gain edits went from 0.761 to 0.207
  (`items_s*.jsonl`, `predictions_run2.json` stage2 note), which is 0.27 of the least-squares value: about a quarter, not a
  fifth. Suggested: "fell to about a quarter".
- NIT: :191-192 call the scan noise 0.0026 "the spread between two scan seeds". It is filed as the sd between two scan
  seeds (`predictions_run2.json`, stage2). Say "the standard deviation".

Spot-checks (jq on `experiments/adjoint_rho/results/items_run2_s*.jsonl`, stage 2), all match:

| Page | Value | Source |
|---|---|---|
| :147-149 | 0.713, 0.453; reach 0.97, 0.51 | x2 E_b_K 0.71268, R 0.9678; rebuilt E_b_K 0.45307, R median 0.5098 |
| :150-151 | new 0.0031, 0.013; reach 0.28, 0.44 | new_few 0.00314, R 0.2796; new_thin 0.01339, R 0.4395 |
| :167 | rebuilt c 0.019 to 0.19, five matrices | five per_matrix entries, c min 0.01893, max 0.18978 |
| :168-169 | remainder 60 to 0.40, cross -131 to -0.54 | x2 shares_of_dW2 outside2 0.4004, cross_out2 -0.5381 |
| :173-174 | held set: new_few 0.0050, random_kl 0.0027 | E_b_over_E268_K 0.00503, 0.00271; x2 and rebuilt equal to E_b_K |
| :188 | identity swap max change 5.1e-5 | stage 0 identity:x2 max_dev 5.15e-5, flips 0 |
