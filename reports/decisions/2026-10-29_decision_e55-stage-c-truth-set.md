# Decision pending: what Stage C of the reuse line is scored against (2026-10-29)

Card id `decision-e55-truth-set`. Drafted by reuse-lead from `reports/analyses/2026-10-29_analysis_e55-stage-a.md`
(Stage A, Amendments 1 and 2, each independently checked: `reports/reviews/2026-10-29_review_e55-stage-a-analysis.md`,
`..._review_e55-amend1-analysis.md`, `..._review_e55-amend2-analysis.md`). Advice from Fable is marked as Fable's.
Every choice here is yours.

## Card text

**Flag:** Soon. **Label:** Question for you · reuse · new 2026-10-29.
**Run line:** Nothing runs on the reuse line until this is decided (Stage C waits on it; Stage B waits on
`decision-e55-plan` item two).

**Heading.** Stage A gave us only two fine-tunes that clearly reuse the old count. What should the methods in Stage C
be scored against?

**What we found.** We trained a small model that counts the letter a, then fine-tuned it to count b, eleven ways.
Two checks that use no decomposition say, for each fine-tune, whether its new answer runs through the old count.
The checks pass their controls. But they call only two fine-tunes "reuses the old count" (both are early, at 1,000
steps, and trained mostly on rows the old count already answers), nine "does not", and five "neither". The plan
expected about six, eight and two. Stage C's pass bar for eleven labelled fine-tunes is ten right, so a method that
always says "new" already scores nine: the test would mostly ask whether a method gets those two right.

The five "neither" fine-tunes are not noise. Reading their answers row by row, three of them (the all-spurious arms
later in training) answer a fixed rule of two of the three counts, not of any one count: one answers the midpoint of
count(a)+1 and count(b), another a point one third of the way between them. Filed in advance, the one-third rule held
on 20,000 fresh rows, but those rows repeat the same pairs of counts, so it is shown stable, not general. The test on
new pairs of counts could not run: the base model itself cannot count a outside about 10 to 17. Because the three
counts always add to 40, answers alone cannot show that count a is one of the two counts used.

### The question: how to build Stage C's ground truth

- **Option A: score the eleven labelled fine-tunes as filed.** Free, ready now. Its result is mostly about two
  fine-tunes; report it with that caveat.
- **Option B: add a third class**, "answers a rule of two counts, not of the old count alone", holding the three
  all-spurious states above, and report what each method says about them, without a pass bar (it is not
  reuse or new by the checks). Free; it makes the "neither" states informative instead of dropped.
- **Option C: first add more fine-tunes that should reuse the old count**, with predictions filed before they run:
  more seeds of the all-spurious arm saved at 1,000 to 2,000 steps (it leaves the old count by 3,000), and a
  fine-tune whose new task is the old count under a new prompt (Fable's suggestion; a positive control more than a
  test). Free on the cluster, about an hour of work and runs; it would bring the "reuses" side to perhaps six to ten.
- **Fable's advice** (before Amendments 1 and 2): run the diagnostics first (done), then Stage C on whatever is
  labelled, and report the nine-to-two imbalance as a finding in itself: on this grammar, a fine-tune that learns a
  new count does not run it through the old one. Put option C on a card rather than run it unasked.
- **Lead's recommendation: C, then A with B alongside.** Two "reuses" anchors cannot score a method; a few more
  seeds at 1,000 steps are cheap and use a recipe already shown to work. B costs nothing and keeps the most
  interesting states in view.

### Smaller, no rush
- **Fix the new-pairs test?** Only five pairs inside the base model's range separate the one-third rule from the
  others, all at the edge. The lead recommends dropping it.

## Links
Report and notebook: `reports/analyses/2026-10-29_analysis_e55-stage-a.md`, `notebooks/E55_pcfg_anchors.ipynb`.
Plan: `reports/plans/2026-10-29_plan_reuse-stage1.md`. Predictions: `experiments/pcfg_anchors/results/predictions*.json`.
Earlier card: `reports/decisions/2026-10-29_decision_e55-plan.md`.
