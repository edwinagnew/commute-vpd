# P1.24 closed half-calibrated: which one new cell should test the causal-rule-retention definition next?

**Question.** A candidate definition of "reused" against "new", causal-rule retention (tested here, not adopted), was
calibrated on three edits of the 67M model whose machinery we built by hand. Two of the three read as they should. The
third could not be built in two runs, and both runs of the cell are spent. Which one new cell should run next?

**What the definition says, in plain words.** Take the original model and an edited copy, and split each into units at
one declared level (here: single MLP neurons, 12,288 of them; or attention heads plus whole MLP blocks, 28). Each unit
has a rule: what it writes, given what it reads. A unit is "retained" to the degree its rule is unchanged, scored from 0
to 1. For a change in behaviour, the reading is the average retention over the smallest sets of changed units that still
produce that change. Low means the behaviour runs on new machinery; high means old machinery, perhaps turned up or down.

**What was read** (two runs on RunPod, about $3.45 in all; page `reports/analyses/2026-11-17_analysis_p1-24-rule-retention.md`).
- **Unchanged copy of the model:** retention 1 on all 12,288 neurons and all 28 heads and blocks, as it must.
- **Circuit turned up:** 8 neurons in one MLP layer carry most of the answer on the copy task. Doubling their output
  read 0.80 per neuron and 0.906 at the heads-plus-blocks level, both above the "high" bar of 0.7. Note that louder
  reads 0.80, not 1: the score is sensitive to size.
- **Circuit replaced by different neurons computing the same values: not built.** The circuit's output was set to zero
  and fresh neurons were trained to reproduce its effect. In 7 fits across the two runs, those that fit well (within
  about 1% of the effect) always copied: one new neuron tracked a circuit neuron at correlation 0.996 to 0.9996. Those
  pushed away from copying lost the fit (44% to 78% of the effect unexplained). One new neuron sat along the shared
  input direction of five of the circuit's eight neurons, and the fit seems to need it.
- So the definition was never shown a replaced circuit: it is neither confirmed nor refuted, and no fine-tune got a
  verdict. The evidence is one fit seed per run, one circuit, one task.

**An open question, before the options (no recommendation).** The premise check's question, quoted exactly: "If a
fine-tune's new units reproduce an old circuit's outputs, what should the definition call that, and what observation
would decide it?" Read at the neuron level the definition would score such units zero, since they have no partner in the
original model. A copy test like ours would call them old machinery. These are different answers, and choosing between
them changes the definition, which only you do.

**Options** (each needs a new one-run cell)
a. **Build the replaced item by fine-tuning (recommended).** Knock the circuit out, then fine-tune all weights of the
   three later MLP layers to recover the doubled-circuit model's outputs. Then run the copy check over every unit that
   changed, not only added neurons, and score the item. It tells us whether a realistic update ever produces the same
   values on different machinery in this model, and gives the definition its missing test item if it does. About one
   short pod run, a few dollars; the fine-tune and scoring reuse existing code.
b. **A hard constraint.** Strip from the new neurons' input weights every component along the circuit's input
   directions, so copying is impossible, then refit. It tells us whether the effect can be produced off those directions
   at all. The fit may simply fail, as the two heaviest penalties did. Cheapest: about a pod run of similar size to the
   last, $1 to $2.
c. **Accept a moved copy as the replaced item.** Score the fit that copied one circuit neuron. It tests only whether the
   score notices a circuit that moved to new neurons, not whether it separates new computation. About $1.
d. **Read the five trained matched pairs on the two built items only.** Gives readings on real fine-tunes now, with the
   replaced case still uncalibrated, so any low reading could not be told apart from the definition being unable to
   see it. About $2.

**Recommendation.** Option (a). It is the only one that answers the open question with an observation: it shows what a
realistic update does when it must rebuild a circuit's effect, and a copy found there would be an answer, not a failed
build. Options (b) to (d) either may fail the same way again, or leave the replaced case uncalibrated.

Evidence: `reports/analyses/2026-11-17_analysis_p1-24-rule-retention.md`; figure
`experiments/rule_retention/figures/replaced_fits.png` (every try at the replaced item; none reaches the corner where
the new neurons fit and do not copy).

Names in the reports: "replaced circuit" is the third built edit; "retention" is the 0-to-1 score above.
