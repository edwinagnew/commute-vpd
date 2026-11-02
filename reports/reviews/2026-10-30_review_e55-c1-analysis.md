# Independent check of the E55 Stage C1 and Amendment 5b analysis

Reviewer: a fresh agent that wrote neither the code nor the analysis, 2026-10-30. Report checked:
`reports/analyses/2026-10-30_analysis_e55-c1.md`. Verdict: **sound with fixes**. Every filed prediction is scored
correctly against its filed rule; the claims follow from the data; nothing post hoc is presented as filed. The fixes
were wording errors, none of which changes a verdict.

## What was checked

- **Code against the filed rules.** `src/commute/revival.py` (`prune_label`, `prune_control`, `revival_z`,
  `reverse_label`, `steer_label`, `pass_bar`, `score`) and `experiments/pcfg_anchors/c1.py` (`truth_from`,
  `truth_sets`, `per_run`, `per_stream`, `nuisance_labels`, `scores`) against `predictions_c1.json` rules, edge
  cases E1-E14 and lead decisions (a)-(d). They match: inclusive ties, sd floored at the binomial sd of the random
  mean, a cap counting as 10,000, void when either gain is not positive, bar raised to majority + 1, nuisance
  agreement counted on the instrument's own non-void states, margin 2 (all) and 1 (per run), per-run vote at the
  latest labelled save (R at 10,000 for steering, as lead decision (b) says).
- **Labels recomputed from the raw readings** in `results/c1.json` with jq (script
  `~/commute-data/scratch/reuse-lead/c1-check/prune.jq`): the pruning gain and sd at both K for all 36 LOW-based
  states, the reverse step counts against scratch, and the steering G and S. Every recomputed label equals the stored
  one, and the per-state table in the report matches to its rounding.
- **Scores recomputed by hand** from the labels, the truth and each state's fine-tune lr (`a2_<arm>.json`,
  `amend5.json`, `amend5b.json`):
  - Every state: pruning 20/26, reverse 17/26, steering 21/27 (R at 1,000 void). Nuisance labellers on pruning's and
    reverse's 26 states: short 18, slow 19, so 21 needed. On steering's 27: short 18, slow 20 (R at 10,000 adds a
    short miss and a slow hit), so 22 needed. Binomial bars 18 (n 26) and 19 (n 27). All as reported.
  - Per run (17 runs; 18 with R for steering): pruning 14, reverse 12, steering 16; short 13, slow 10 (11 with R);
    needed 14. Pruning and steering pass, reverse fails. As reported.
  - Per stream, described (13; 14 for steering): pruning 10, reverse 8, steering 12; needed 11, 11, 12. As reported.
  - Lr strata and class-B label counts, all five labellers: as reported.
  - Post hoc steering without R: 21/26 against needed 21; per run 16/17 against 14. As reported, and labelled post hoc.
- **The margin rule.** Agree >= best nuisance + margin, as filed (E12). Note: the filing's table expected steering to
  need 23 (slow 21 of 28); R's void at 1,000 steps removed a state the slow labeller gets right, so at the actual n
  it needs 22. The rule is "at the actual n", so 22 is correct; one sentence added to the report saying so.
- **No overlap of the raw readings** on the 26 scored states: pruning gain at the stronger K 0.354-0.748 vs -0.181 to
  0.082; it also holds at K = 5 alone (>= 0.244 vs <= 0.057) and at K = 15 alone (>= 0.143 vs <= 0.082). Reverse
  steps over scratch 0.075-0.152 vs 0.180-0.609. Steering S 0.836-0.984 vs 0.369-0.669. Confirmed; the report labels
  it exploration, fitted, not filed.
- **Controls.** Importance-pruning drops 0.0573 (229 rows) and 0.2039 (816 rows) against 3 sd of 0.00075 and 0.00087;
  Alg. 3's direction lowers LOW by 0.040 and 0.062; scratch 2,300 steps at 1e-5, cap (0.62) at 1e-6. As reported.
- **Amendment 5b.** Classes, B and P per save from `amend5b.json`, seed 0 from `amend5.json`, bars from
  `predictions.json` a2_bars, W0 controls, shortcut and off-label reads. P1 met 3/3 as reported; seed 3's margin
  0.027 is under 2 binomial sd (0.032).
- **Figures** (`c1_scores`, `c1_prune`, `c1_reverse`, `c1_steer`, `amend5b_saves`) against `plot_c1.py` and the
  data: each shows what the text says, with its reference on the same axes (bar and needed ticks beside the nuisance
  bars; the random-draw zero line and 3 sd ticks; scratch and half-scratch lines; the 0.5 line and the do-nothing
  zero; seed 0 and LOW beside seeds 1-3). Not re-rendered: `plot_c1.py` reads only the JSON, and every plotted
  number was recomputed above.
- **Notebook section** (cells 35-39 of `notebooks/E55_pcfg_anchors.ipynb`) and **the three PENDING lines** against
  the report: the same numbers and verdicts. The card's "$10-30" for C2 matches the plan's cost table; "at most 7 of
  26" (26 - 19) is right.
- **Post hoc presented as filed.** None found. The margin, nuisance labellers, per-run score and the G < 0.01 flag
  were all filed or decided before any run (`lead_decisions`, 97b7183 and 8de9324); the per-stream score and strata
  are called described; the no-overlap reading and the score without R are called post hoc.

## Findings

1. Report and notebook: "the random draws removed none" on the LOW pruning control. At K = 15 three of 20 draws
   lost 1-3 rows (mean 0.99966). Fixed to "at most 3 rows".
2. Report, pruning: "removing five units brings back..." for every "reuses" state; the four slow natural-row seeds at
   1,000 steps read best at K = 15. Fixed.
3. Report, Amendment 5b: "At every save the count-b patch share sits between the bars". At 500 steps it is under the
   lower bar ("not through"). Fixed; the conclusion (count b not "through", so reuses) is unchanged.
4. Report, Amendment 5b: "Seed 1 at 2,000 steps misses the behaviour bar by 0.004" sat in the letter-swapped
   paragraph and read as a swapped-row result; it is the main check (B_a 0.496 vs 0.4999). Fixed: "On the main check".
5. Report, Amendment 5b figure: LOW's patch point sits at 0.96 in the "through" band with no explanation. Added that
   LOW answers the count of a itself.
6. Report: "The new states sit at 0.2175"; two read 0.215 and 0.217. Fixed to "about 0.2175".
7. Notebook: R's gain "against about 10 elsewhere"; the other states read 6.2 to 10.5. Fixed to "6 to 11", as the
   report says.
8. Report: the filed expectation that steering needs 23 differs from the scored 22, unexplained. One sentence added
   (finding under "The margin rule" above). Not an error in scoring.

No change to the PENDING lines: they agree with the report. No predictions file touched.

## Skipped

- Re-rendering the figures: they would overwrite the committed PNGs for no gain; numbers checked from JSON instead.
- `check.py` style flags (64 before, 65 after) were left as the author set them; the one added is the sentence on
  the needed 22.
