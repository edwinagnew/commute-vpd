# Advice: what the size test shows, and what to do next (Fable, 2026-11-02)

*Advice, not a decision. Sources: `experiments/qwen_recipe/results/size_verdict.json`, `size_select.json`,
`train_gold_1e-5.json`, `train_a1_*.json`, `train_a2_6e-6.json`; the rule in `results/predictions_size.json`; the plan
`reports/plans/2026-11-02_plan_e61-direct-share-size.md`. Numbers are provisional until the independent check reports.*

## What the result shows

The filed line reads C, and it stands: the small gold state (step 8, size 0.57) has a direct share of 0.026, the same
as the two random changes at that size (0.023 and 0.030), and the rule says any gap at most 0.01 is C. But the rule's
label for C, "size explains the direct share", says more than the data do. At the big size the gap is 0.038 on one
axis and 0.045 on the other, four times the floor, with intervals of about 0.003, and a second gold run read 0.09 at
a larger size. So the true statement is: **a fine-tune's excess over random appears only after the fine-tune has gone
some way (between steps 8 and 100); at step 8 there is none.** Size alone cannot be said to explain the share, because
the two randoms at the big size were scaled to it and still read 0.026 and 0.039. What the line was designed to need,
an excess at both sizes, is not there.

What it does not show: one seed per kind; maths rows only; two random nulls, both scaled along a straight line in
weight space; a linear interpolation between two reads of each random; the share at small size has a small denominator
and every change reads 0.02-0.03 there, so a floor of the measure itself is not ruled out; and nothing here separates
"already there" from "new". Step 8 is a state on the way (loss drop 1.86 of 1.99, answer length 0.81 of W0's against
0.48 at the end), so "the excess needs more training" and "the excess needs a bigger change" are not yet told apart.

## The one cheap read worth doing

The 16-row sizes in `size_select.json` plateau: steps 20, 30, 50, 100 read 1.17, 1.26, 1.27, 1.30 (K00 − K0), while
training continues (held-out loss 0.556 → 0.510, answer length still falling). The share at step 100 is 0.077. Reading
**step 20 and step 50** (saved in `results/large/train/gold_1e-5/`, about 26 min each, $0, cluster, same code path
as Job S's `state` stage) separates the two readings above:
- step 20 and 50 near 0.077: within gold the share is a function of size; the excess over random is a property of the
  big change, and C is fully earned.
- step 20 well under 0.077 (say under 0.06) with step 50 between: the share keeps rising while size does not, so it
  tracks something training does beyond moving the outputs. That is the only result that would make the reference
  comparison worth its hours.
This is exploration, not a change to the filed verdict: the verdict is scored and closed; the new reads go in the
analysis as the cause-of-miss paragraph, with the expected values above written down before the job runs. Step 12
(size 1.02) could be added as a third point for the curve; not needed for the question.

## Options

1. **Stop and write up now.** Runs nothing. Says: the switching reading cannot separate anything; the direct share puts
   a finished maths fine-tune 0.04 above random at matched size, and a part-way state at random level; size alone does
   not explain it, and no reading separates "already there" from "new". Cost: writing. Next: GSM8K and RL readings are
   description only.
2. **The two saved gold states first, then write up** (recommended). Runs step 20 and step 50 through the same reading,
   about 1 h, $0. Tells you whether the excess grows with training when size has stopped growing. Next: write up
   either way; if the share rises on the plateau, option 3 becomes a live question for the user.
3. **Compare the template and invented-facts fine-tunes at equal size.** Runs about 3.5 h, $0; both are trained and
   learned (Job R). Tells you whether the known "already there" and "new" cases differ in direct share at equal size.
   The catch: the template moves the maths outputs by a KL of 0.10-0.13 from W0 (training record), under half the
   step-8 gold state's 0.26, where even gold reads at the random level. Matched to the template, both would be expected
   to read the 0.02-0.03 floor, and a null there would be the measure's floor, not an answer. Read unmatched (template
   at 0.1, facts at step 300, KL 0.50), any gap is size again. Next: only worth it if some state of each lies where
   gold shows an excess, which the template never reaches.
4. **A second gold seed at steps 8 and 100.** About 40 min of training plus 1 h of reading, $0. Tells you the
   seed-to-seed spread the floor 0.01 stands in for. Next: tightens the write-up's intervals; changes no reading unless
   the spread is over 0.03.

## Recommendation

Option 2, then write up under the filed C, worded as above. Do not run the reference comparison: the size test says
the direct share is blind below about size 0.6, and the template's size lies under that, so the comparison it was
meant to be cannot be read with this measure on these fine-tunes.

What would change it: the independent check moving gold's big-state share toward 0.04 or the randoms toward 0.07
(then C is earned as labelled and option 2 is skipped); the template's K00 − K0 on 128 rows coming out near 1 rather
than the 0.2-0.3 its KL implies (then option 3 is readable, and the step-20/50 result decides whether to run it); or
step 20 reading well under step 100, which turns the write-up's closing line from "closed" to "one measure still
moves with training, untested against the reference pair".
