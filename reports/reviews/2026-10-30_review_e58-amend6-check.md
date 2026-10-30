# Review: independent check of the E58 Amendment 6 write-up (commit 4068936)

*2026-10-30, fresh reviewer (wrote neither the code nor the analysis). Scope: commit `4068936`, i.e. the "Amendment 6"
section of `reports/analyses/2026-10-30_analysis_e58-open-pair.md`, the changed parts of
`reports/analyses/2026-10-30_analysis_regating-synthesis.md`, the A6 cells of `notebooks/E58_open_pair.ipynb`,
`experiments/open_pair/plot_amend6.py` and both `a6_*.png`, the `experiments/TRIED.md` row and the
`docs/research_log/PENDING.md` line. Data read with `jq`: `experiments/open_pair/results/amend6.json`,
`predictions_amend6.json`, `amend3.json`, `amend4.json`, `amend5.json`; log `cluster/logs/commute-e58-open-pair-46489.out`;
plan § "Amendment 6".*

## Verdict

**Needs fixes, small.** Every number but one matches the JSON, both filed rules are scored as filed, and the
predictions were not edited after the run. The one wrong number is a surprise line. The main issue is scope: the
write-up says the result shows r reads "more than the prior". The data show less than that. A prior cut between 3.7e-5
and 5.9e-5 still sorts all six read maps into r's two bands. What the result excludes is a smooth prior-only trend,
the post-hoc line. The first-letter cell is correctly called untested. The granularity limit sits beside the claim in
the analysis's blind-to list, the notebook and the synthesis.

## Provenance and scoring: checked, no error

- Predictions filed `a946868` (17:56:34), amended `5643589` (18:07:58) after the code review. The job's first log line is 18:09:16
  (`[host]`). The amendment is recorded in the file (`amendment_2026_10_30_code_review`). `sha256sum` of
  `predictions_amend6.json` = `9af9f9ef…`, equal to `amend6.json` `predictions_sha256`. No commit touches the file after
  `5643589`. `git_head` in `amend6.json` = `5643589`, `git_status_code` empty.
- W₀ control: `zero_max_abs` 0 on both maps; prior 3.726e-5 in [2e-5, 2e-4]; 8.22e-7 < 1e-5. Pass. Harness control
  r 0.32412633847537903 equals `amend4.json` exactly. Pass. Both maps used `cipher_pool`, so PM is scored.
- PM as filed: r30 0.3729 / 0.3806 / 0.3817, all < 0.5, so the class is `cipher_band`. Nearest gap to the cut is 0.118,
  above 0.0452, so `near_cut` is false. Correct.
- FL as filed: 3 labellable step-30 reads ("neither", x finite). The x − x_null values are 0.0405 / 0.0633 / 0.0825,
  all ≤ 0.3, so `no_carry`. Correct.
- Spearman 0.8857 = 1 − 6·4/(6·35), i.e. n = 6 maps (first-letter has no r). The write-up says "six maps". Correct. The
  filed "reported" item said seven maps; reporting six is right because first-letter has no r.

## Numbers checked against the JSON

All correct: priors 1.4e-6 / 1.0e-5 / 3.7e-5 / 5.9e-5 / 1.3e-3 / 1.3e-2. Factor 1.586, so "about 1.6". The shuffle's prior is
3.57× lower, so "3.6". The prior-only line reads 0.6929 at the star (slope 0.3904 per decade through the shuffle mean
0.4770 and the copy mean 0.7711); the star mean 0.3784 sits 0.3145 under it. Every table row in the analysis and the
notebook (r at the gate and at 30, ratio, x, G_spec 19.0-20.1, cipher's 16.1-18.1) matches. The first-letter table
(gate steps 12/13/10; x, x_null and their difference at the gate and at step 30; G_spec 16.7/17.0/16.3) matches. Seed range
0.0088, "0.009". "0.12 under the cut" and "0.10 below the shuffle" both match.

## Errors

1. **`reports/analyses/2026-10-30_analysis_e58-open-pair.md:940`**: "the writer's slope argument missed by about 0.3".
   The writer expected about 0.6, and the star's mean is 0.378, so the miss is **about 0.2** (0.22). The 0.31 is the gap to
   the post-hoc shuffle-copy line (0.69), which is a different expectation from the writer's.
2. **`reports/analyses/2026-10-30_analysis_e58-open-pair.md:820`**: "It first cleared the gate at step 10 on every
   seed". Label reads start at step 10, so the gate step is censored. G_spec was already 5.73 / 6.34 / 5.86 > 5.148
   at the first read. Say "was already over the gate at step 10, the first read step". This also misses the filed
   expectation "gate step 12-16". Not scored, but worth one line in Surprises. The step-10 r of 0.431-0.454 sits
   0.046-0.069 under the cut, at the edge of the 0.045 noise floor. Say so beside "cipher_band too, 0.43 to 0.45".

## Overclaims

1. **Is the prior a threshold or a trend?** The realised prior sits a factor of 1.6 *below* copy's. Fable asked for one
   at copy's prior. Run a single cut on the geometric-mean prior anywhere between 3.7e-5 and 5.9e-5: it puts cipher,
   shuffle and the prior-matched cipher in the low band and copy, elicit and rank-1 in the high band. That is all six
   maps, with no error. So "r reads only the prior" survives as a step function in the prior. What the run excludes is r
   rising smoothly with log prior, i.e. the dash-dot line. The data do argue against a step. The shuffle reads 0.10 above the
   star at a lower prior, so the order is not monotone. A step of 0.39 inside 0.2 decades, with r flat to within about 0.13
   over the next 2.3 decades, is not what a smooth dependence looks like. But a step is not excluded. Affected wording:
   - analysis `:764`: "It rules out 'r reads only the prior': if it did, this map would read near copy". That holds
     only for a smooth dependence on the prior. Say it rules out r following the prior smoothly, not a cut between
     the two priors.
   - analysis `:801` and synthesis `:111`, also the figure legend "r if it tracked the prior only": the line is *one*
     prior-only model, fitted post hoc through two points. Say "a smooth prior-only trend" and add that a cut
     between 3.7e-5 and 5.9e-5 still fits every map.
   - synthesis `:13` ("on that one pair the reading is more than the model's prior") and the analysis/notebook "The
     answer": add "more than a smooth function of the prior".
2. **Synthesis `:138-139`: "That ranking defines the split, so it is the baseline r must beat. On one map pair r beats
   it."** This is wrong as worded. The ranking (whether the target is W₀'s own guess for the key) is exactly what the
   prior-matched cipher keeps at "not own guess". Its values were admitted only at rank over 100, and r agreed with
   that ranking again. What r beat is W₀'s *probability* of the target, not its ranking. A W₀-only baseline built
   from rank or key-specific preference still predicts every band; it was neither matched nor measured. Reword it to: "On one map pair
   r reads more than a smooth function of W₀'s probability of the target. It still agrees with W₀'s own-guess ranking,
   a baseline that never looks at the decomposition."
3. **"Kind" means two things.** At analysis `:858` and `:898`, synthesis `:112` and in the notebook, "r separates
   kind" means own-guess link versus none. At analysis `:806` ("the kind of word moves r on its own") and in the PM
   `blind_to`, "kind" means the token pool. In one paragraph a reader cannot tell which. Use "own-guess link" for the first
   and "word pool" for the second. Also, the headline "separates kind" sits next to "this does not say which of those
   [kind, carry, trained gain] r reads". Copy differs from the lookup in link, carry, trained gain and being a rule, so
   the supportable headline is "r separates copy from a no-link lookup at a prior 1.6× lower". The notebook's caveat
   names carry only. Add trained gain (copy's G_spec 25.4-25.7 against the lookup's 19.0-20.1).
4. **TRIED row and PENDING line** carry "r separates kind at a near-equal prior" without the threshold caveat. When the
   wording above is fixed, update both. The TRIED row also lists only `a6_prior.png`; add `a6_first_letter.png`.

## Scope points that are right

- One map pair and one prior level, stated in each file. Provisional until the independent check, stated.
- First-letter is "untested / stays empty", not negative, in all four places. The analysis and the synthesis's
  "How compelling" give the null-corrected 0.04-0.08 against the 0.3 bar.
- The blind-to list names the per-key spread of the prior, the pool, three seeds, one split and one decomposition, plus
  the granularity limit.

## Figures

- `a6_prior.png` matches the JSON. The first-letter triangle is drawn at r = 0.2, a placeholder (`plot_amend6.py:142`).
  It is labelled "r not read", but on an r axis it reads as a value. Move it to a strip below the axis or to a text
  note.
- `a6_first_letter.png` matches the JSON and is clear.

## Not run (by this review)

- No re-computation of r, x or x_null from saved states (none saved, by design); the scoring code in
  `src/commute/open_pair.py` was not re-read, only its outputs checked against the filed rules.
- Copy's per-key rank of its target was not looked up. It would say whether a rank baseline separates copy from the
  prior-matched cipher (overclaim 2).
