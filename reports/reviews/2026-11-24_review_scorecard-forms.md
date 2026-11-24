# Scorecard reading forms: which reach the paper (redteam-lead, 2026-11-24)

The reading-refactor audit (`reports/plans/2026-11-25_plan_reading-refactor.md`) found four readings computed in more than one way, plus two smaller differences. This page checks each against the paper on origin/v2. Two change a figure and a call (MUST). Three need only a stated unit or form (SHOULD). One does not reach the paper.

baseline-lead's note holds for known_main, known_scorecard and known_appendix, which draw one form per reading. It does not hold for Fig 3a or Fig 4 (`paper_fig3a_share.csv`, `paper_fig4_share_vs_prior_steering.csv`), which read `method_table/results/final.json`.

## MUST
**1. The re-gating share comes in two units on one axis.** Fig 3a and Fig 4 take their share rows from final.json. Most rows are in log-odds. e123 and e125 are `real_vs_twin/stage2 .share.contrast` (Fig 3a: e125 -0.090, CI -0.155 to -0.018; e123 -0.018).
- Checked from `experiments/real_vs_twin/results/stage2.json`: e125 in log-odds reads real 0.160 against twin 0.191 (gap -0.031), and the seed ranges overlap (real 0.110-0.237, twin 0.175-0.219), so there is no call. In the contrast unit the ranges are disjoint (real at most 0.130, twin at least 0.148), so the call is twin higher. The unit changes the call.
- The text's tally (`sec5_baselines_scorecard_caveats.tex:19-20`, "real higher on one, twin higher on one, no call on two"; tab:scorecard twins column, `appD_built_pairs_more.tex:27`) matches log-odds. Fig 3a draws e125 clear of zero, so the figure and the text disagree.
- Fix ($0): take e123 and e125 from `stage2.json .states[].share.lo`, then regenerate both CSVs.

**2. Steering mixes three forms on the Fig 4b x-axis.** final.json's steering value comes in three forms:
- E143 pooled (`holes_fill .pooled.share`) on most built pairs;
- per-key on e54_matched (-0.237) and e54_own (-0.120);
- per-key in the contrast unit on e123 (-0.506) and e125 (-1.087).
- e62 uses a fourth source, E135 `steering_copy_lookup .pooled.share_lo`.

Pooled gives e54_own +0.114, so its sign flips. On the e123 twin, pooled says real higher where per-key says twin higher. known_scorecard already plots pooled for e54, so Fig known and Fig 4 disagree on the same pairs.

Tab:scorecard shows both forms, which is fine. Three sentences are not:
- `appE_methods_detail.tex:179` still states the wider-gap rule.
- "reverses three to five of nine" (`sec5_3_mechanism_vs_prior.tex:88`, tools table `sec5_results.tex:180`) holds for per-key only; pooled reverses 2.

Fix ($0): rebuild the Fig 4 x values from three pooled sources:
- `holes_fill/results/stage1.json .steering.pooled.share`;
- `refit_96key` pooled for the e58 pairs;
- `real_vs_twin/stage3 .pooled.share.lo` for the twins.

Then re-read e62 under the E143 rule, or mark it. Drop the wider-gap sentence, and recount the reversals.

## SHOULD
**3. Prior.** Fig 4a, known_main, prior_mech_2x2_cued and E199 are all log10, and the Spearman 0.02 (`sec5_3:19`) is all log10. realistic_mirror's "Base prior" is W0's mean natural log-odds, used only within its own 12-task τ. Its caption (`sec5_results.tex:151`) calls these "the readings of Figure known", and the axis has no unit. Fix: label the unit (natural log-odds), or recompute log10 GM.

**4. bf16 batch dependence.** e53's share moves by at most 0.0025 across batch sizes, against gaps of at least 0.39. Fix: state the batch size in appE.

**5. KL forms.** The procedural partials (`appI:145`) use KL(W0‖F) throughout, and the built-pair regression (`appI:51`) uses KL(F‖M) throughout. Nothing pools the two forms. But appI:51 says "KL from the base model" for KL(F‖M). Fix: name the direction in both places.

## Not in the paper
**6. Size units.** final.json's summed_norm and norm2 reach no figure or sentence. The size columns in known_scorecard, known_appendix and appD:37 are relative size. The "matched" series come from `size_matched/summary.json`, which matches relative size on both sides. Tab:scorecard's size row (0/7/2) is from a re-derived source (final.json reads 6 opposite, 3 no call; source not traced). Either way no pair agrees, so `sec7_discussion.tex:49-50` stands.

## Not run
- I did not trace which file tab:scorecard's size row comes from.
- No re-read was done. All fixes above are from saved files, for the figure owners.
