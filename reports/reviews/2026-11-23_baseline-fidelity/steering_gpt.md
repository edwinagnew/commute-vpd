# Steering fidelity: GPT adversarial reviewer (2026-11-23, condensed faithfully; HEAD e894f1ed)

Verdict: needs-attention. The code implements a defensible fine-tune activation-shift recovery share, not ActAdd, CAA or Venhoff's hybrid. No ship with the current attribution or figure label.

## Fidelity to the cited methods
- **Blocker, ours: "This is the test of Venhoff" is too strong.** (main.tex:930; fair_steering.py:44; holes_fill/run.py:162)
  - Venhoff (arXiv 2510.07364 v4): thinking-model rollouts teacher-forced through the base; disagreement positions; SAE-derived categories; jointly optimised category vectors with a coefficient MLP; ~37% depth; best of 3 on validation; reads the recovered fraction of math accuracy. v1 differs (bias vector, perplexity-selected coefficient).
  - Ours: same-input tuned-minus-base block-input means at one predictor position, one scalar and block, a log-odds ratio on fixed windows. A high share can come from restoring the known target at the site.
  - Fix: "borrows the gap-recovery comparison from Venhoff et al.; our intervention and outcome differ", and pin the version.
- **Major, ours, partly disclosed: ActAdd supplies the addition operator, not the vector or the evaluation.** (main.tex:926; steering.py:60,147)
  - ActAdd: contrast prompts in one model, per-token differences, generation evaluations.
  - Ours: a two-state mean at the predictor position only, scoring its immediate logits.
  - Fix: say "inspired by activation addition", not "ActAdd baseline".
- **Major, ours, partly disclosed: CAA supplies the averaging idea.** (fair_steering.py:33,44,60,72; real_vs_twin/run.py:96,299)
  - CAA: within-model contrasts at the A/B letter, applied through generation, both signs, norm-standardised, layer sweep at ±1.
  - Ours: raw norms, nonnegative strengths, a joint block x strength search, one position.
  - A per-key vector bank can act as a lookup table of the fine-tune's changes.
  - Minor: the Ward citation (main.tex:931) is correct and is context only.

## Manuscript-to-code departures and selection
- **Blocker, ours: the label and caption say strength 1, but the figure is tuned.** (appD_built_pairs_more.tex:98-112; figures/src/make_figures.py:99-108; figures/data/paper_fig4_share_vs_prior_steering.csv:31-34)
  - The ordinary points are tuned; only the star is E199 strength 1.
  - Mid/cipher at equal size is plotted at -0.237; E149's strength-1 value is +0.038.
  - The tuned nine-pair Spearman is -0.30 against a recorded strength-1 -0.10 (share_gap_sizes/results/gaps.json). The all-nine prior-plus-steering claim is strength-1 only; tuned gives 6 agree / 3 conflict.
  - Fix: relabel the figure, or regenerate it from strength-1 data.
- **Major, ours: copy-lookup uses held-out fit windows, and re-gating is read on different rows.** (steering_copy_lookup/run.py:48,168,203,285,313; share_untrained)
  - E135 fits on the even 100 of E62's 200 windows and scores on the odd 100. The table uses E136's re-gating on all 200 (0.691 vs 0.686 on matched rows; steering 0.412).
  - Fix: use the saved matched comparison, or disclose the exception.
- **Major, ours: selection at three levels; the last uses read outcomes.** (holes_fill/run.py:105,162,182; method_table/plot_final.py:60-134; steering_copy_lookup/run.py:128; size_matched/run.py:143,150,233)
  1. Vectors come from training windows.
  2. The grid picks the maximum mean rise over blocks 0-3 x strengths {0, .5, 1, 2, 4, 8}, on the same training rows that built the vectors. All 94 E143 winners pick strength 2, 4 or 8.
  3. The scorecard picks the largest absolute gap between forms on the read rows. On mid/cipher at own steps, pooled reads +0.114 and per-key -0.120: a 0.006 advantage changes the sign.
  - Size matching refits and reselects.
- **Major, ours: fixed strength is not fixed layer, and the "apart" rules differ.** (steering_reversal; prior_steering/rules.py:33-48; holes_fill/rules.py:29)
  - E146 reselects the block at strength 1.
  - E149's vote needs a 0.10 gap above the within-side spread, while the scorecard uses strict nonoverlap. Under nonoverlap, own-step mid/cipher would oppose (reuse max 0.722 < new min 0.751), so the joint vote tracks by prior alone.
  - The scorecard's rule calls small gaps the filed experiments did not (96-key shuffle +0.043, class twin -0.074).
  - Fix: disclose the distinct voting rule.
- **Minor, ours: define the scale and controls.**
  - The share is a ratio of mean rises with no clipping, and the gain floors differ between experiments.
  - Random vectors, cosines and off-position KL are diagnostics.
  - E244 runs no steering.
