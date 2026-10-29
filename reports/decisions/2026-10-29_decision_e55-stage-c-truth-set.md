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
new pairs of counts could not run: on the rows built for it, even the base model fails its own count. An
independent check traced that to how those rows were built (letters swapped inside the counted window), not to a
counting limit. Stage A's two checks read natural text only, so its labels cannot carry this. But the all-spurious
arms were trained on rows built the same way, so whether their early "reuses the old count" reading means LOW's own
counter is what a free check (Amendment 3, being run) tests, with natural rows in the same pairs. Outside the
pairs it was found on, the one-third rule has 0 of 2 rows in its favour so far. Because the three counts always add
to 40, answers alone cannot show that count a is one of the two counts used.

### The question: how to build Stage C's ground truth

- **Option A: score the eleven labelled fine-tunes as filed.** Free, ready now. Its result is mostly about two
  fine-tunes; report it with that caveat.
- **Option B: add a third class**, "answers a rule of two counts, not of the old count alone", holding the three
  all-spurious states above, and report what each method says about them, without a pass bar (it is not
  reuse or new by the checks). Free; it makes the "neither" states informative instead of dropped.
- **Option C: first add more fine-tunes that should reuse the old count**, with predictions filed before they run:
  all-spurious fine-tunes trained on natural texts only (about one natural text in five already has count(b) =
  count(a) + 1, and LOW gives the right answer on every one of them), several seeds saved at 1,000 to 2,000 steps;
  and a fine-tune whose new task is the old count under a new prompt (Fable's suggestion; a positive control more
  than a test). Free on the cluster, about an hour of work and runs; it would bring the "reuses" side to perhaps six
  to ten, on a recipe where reusing the old count needs no repair.
- **Fable's advice** (before Amendments 1 and 2): run the diagnostics first (done), then Stage C on whatever is
  labelled, and report the nine-to-two imbalance as a finding in itself: on this grammar, a fine-tune that learns a
  new count does not run it through the old one. Put option C on a card rather than run it unasked.
- **Lead's recommendation: C, then A with B alongside.** Two "reuses" anchors cannot score a method; a few more
  seeds at 1,000 steps are cheap and use a recipe already shown to work. B costs nothing and keeps the most
  interesting states in view.

**Amendment 3 (read 2026-10-29) weakens the two "reuses" states.** On the swap-built rows the all-spurious arms
were trained on, LOW's own count gives the right answer on only 45% of rows (all with no swap, 2% at a swap of
three letters), and the early states answer those rows right by 1,000 steps (99%), almost never with LOW's answer. On natural
text their answer runs through a count of a, so they stay labelled as measured. Either they count b on the built
rows (their count of a may still be LOW's) or they strengthened the count of a itself (still the "existing" side of
the question); a patch check on built rows would tell, and has not run. What weakens is the premise that nothing
pushed these fine-tunes to change LOW's route. This is why option C now trains on natural texts only, and why no
method should be scored on swap-built rows.

**The cost of training on natural texts only:** their answers are narrow (of 760 such test rows, 509 have answer 14,
so a constant 14 is right on 67%), unlike Stage A's built rows, whose answers were matched to the natural spread of 10
to 17. Option C's arms would be easier to fit without counting; their predictions would need a bar above that.

## Links
Report and notebook: `reports/analyses/2026-10-29_analysis_e55-stage-a.md`, `notebooks/E55_pcfg_anchors.ipynb`.
Plan: `reports/plans/2026-10-29_plan_reuse-stage1.md`. Predictions: `experiments/pcfg_anchors/results/predictions*.json`.
Earlier card: `reports/decisions/2026-10-29_decision_e55-plan.md`.
