# Analysis: on the 67M, does VPD tell elicitation from a new mechanism? The re-gating line's synthesis (E54, E56, E58)

*2026-10-30, regating-lead writer. Claim wording, evidence split and figure follow Fable's advice
`reports/plans/2026-10-30_advice_e58a4-synthesis-FABLE.md`, adopted by the lead. The record behind every number:
`reports/analyses/2026-10-30_analysis_e58-open-pair.md` (E58 and its Amendments 1 to 7),
`reports/analyses/2026-10-30_analysis_e54-keydependent-elicit.md`, `reports/analyses/2026-10-30_analysis_e56-keydependent-patch.md`.
Figure: `experiments/open_pair/plot_synthesis.py`. Words as in `GOALS.md` → Terms. Provenance at the end.*

On the small model, VPD's main reading separates fine-tunes whose trained answer the original model already guesses from
those whose answer it does not, and it reads a new rule that generalises to unseen keys as new, the same as a plain
lookup. Across eight word maps and three seeds the reading followed the model's own guess, not whether learning carried,
and a lookup at nearly the copy map's prior still read low. For `GOALS.md` this says the reading tracks "the answer is
one the model already gives", which the model's own prior largely gives too. It is not yet evidence that VPD detects
a new mechanism as such: it is one rule on one small model, it awaits an independent check, and a new circuit of old
subcomponents onto answers the model already gives, where the granularity limit bites, is untested.

## The answer

On the small model with trigger-to-word maps, VPD's r reads high when the trained answer is the model's own guess for
each key and low when it is not, whether or not the update carries to unseen keys. A new four-word rule whose words W₀
never gives carried 0.55 to 0.64 beyond its null to unseen keys, yet read r 0.23 to 0.25. That is with the lookups and
within 0.04 of its shuffled twin, which has the same words and no rule. Copy, which carries about as far, reads 0.77.
So r does not read carry, and a new rule that generalises reads as new. The rule is built from directions W₀'s
embedding already has, so the granularity limit did not make r read it as old. The limit is untested where it would
bite: a new rule onto answers W₀ already gives would, on the own-guess account, read high. A no-rule lookup at 3.7e-5
reads 0.38 while copy at 5.9e-5 reads 0.77, so r is not a smooth function of W₀'s prior. A cut on the prior between
those two still sorts all eight maps, and W₀'s own-guess ranking still predicts every band. E58's one comparison with
a steering vector, a reading that never looks at the decomposition, scored steering the better match, though steering
sat on its cut. The claim is provisional until the independent check. This holds on one small model, one decomposition
and synthetic maps only.

## What we were trying to test

The re-gating line asks whether a VPD decomposition can tell a fine-tune that strengthens something the model already
does from one that teaches something new. On 2026-10-27 the user de-prioritised the fixed-shift hypothesis, recorded in
`reports/decisions/2026-10-27_decision_regating-fixed-shift-deprioritised.md`. That work, E18 to E36, is set aside here,
and none of its results are used.

**The task.** Each training window is Pile text, then a rare marker token, then a key word; the model must predict the
key's value next. One set of keys, five maps from key to value:

- **elicit**: a word the original model W₀ already ranks 2nd to 20th after that key;
- **rank-1**: W₀'s own top word after that key;
- **copy**: the key itself;
- **cipher**: an arbitrary word W₀ gives under 1e-4 there;
- **shuffled elicit**: elicit's words, but each trained key takes another trained key's value, so no value is W₀'s
  guess for its own key.

The first three have targets that are W₀'s own guess for each key; the last two do not. A fine-tune trains some keys,
24 or 96, and holds out a quarter of the keys it never sees.

**Two measurements per fine-tune.**

- $x$, the held-out carry: the unseen keys' gain in their own value's log-odds over the map's other values, divided by
  the same on the trained keys. It reads 0 if unseen keys gained nothing and 1 if they gained as much as trained ones.
- r: Γ is VPD's function that sets each subcomponent's causal importance. r is the share of the fine-tune's gain
  recovered when the subcomponents stay as W₀'s and only Γ's importance values change to those on the fine-tuned model.
  It reads 1 if switching existing subcomponents on carries the whole gain, 0 if none of it. The stand-in ratio is a
  companion reading with no ceiling at 1.

The question in these terms: does r follow whether the target is W₀'s own guess, or whether $x$ is high? Since "own
guess" is defined by W₀'s ranking of the target, that ranking alone already sorts the maps; it is the baseline r would
have to beat to show the decomposition adds anything.

## What we expected, and why

The line's working reading, adopted by the lead after E58 and called reading A, was that r reads which subcomponents an
update runs through, not whether what it learned generalises. Two filed tests were meant to score it. Amendment 3
predicted r would stay flat on each map as carry rose from 24 to 96 trained keys, about 50%. Amendment 4 predicted r
would read the same across the three own-guess maps within twice its seed noise; Fable leaned pass at about 70%, the
plan's writer at about 30%. The shuffled map was expected to read like the cipher, about 60%.

## Does r separate maps by the model's own guess, or by how far the rule carries?

![Held-out carry x against r and the stand-in ratio for every E58 map, at 24 and 96 trained keys and both read points, three-seed means with seed ranges, E54/E56 bands and the 0.5 cut](../../experiments/open_pair/figures/synthesis.png)

**Reading it.** By the model's own guess, which is also W₀'s prior ranking of the target. Read the vertical axis of the
left panel. Every own-guess map, green, purple
and blue, sits at r 0.72 to 0.92 at 96 trained keys. Every map whose target is not W₀'s guess for that key, orange and
grey, sits at 0.29 to 0.49. At 24 trained keys the same split reads 0.76 to 0.85 against 0.22 to 0.24. E54 and E56, which
train all 32 keys, read 0.76 to 0.77 against 0.21 to 0.27, on the same sides of the cut. The gap between the bands is at least 0.24 at every read point,
over five times the largest seed range of 0.045. Every map in the lower band also has carry at or under zero. The
plot's lower right is empty: no map has a target W₀ did not guess that still carries to unseen keys.

Now the horizontal axis. Inside the upper band carry runs from 0.10 to 0.84 at 96 trained keys, and from about −0.03
to 0.55 at 24. r is not in carry order. At 96 trained keys the rank-1 map has the highest r at both read points and
carries about a fifth, while the copy map carries over four fifths at lower r. Nor is r unrelated to carry. The elicit
map has both the lowest carry and the lowest r at every read point on three-seed means (at step 10 one rank-1 seed carries 0.19, below elicit's 0.21-0.22), and over the nine 96-key reads the rank correlation
of r with carry is 0.42 at step 30. So r is not blind to which map was trained, but carry is not what orders it there.
The stand-in ratio, right panel, splits the same way: 0.96 to 1.21 against 0.41 to 0.66 at 96 trained keys.

**The shuffled map** is the sharpest single test. It has elicit's words but not their link to each key. At step 30 it
reads r 0.47 to 0.49 over three seeds, only 0.015 to 0.03 under the 0.5 cut, and 0.17 to 0.19 above the cipher. If r read only the kind of
word, it would sit with elicit at 0.72 to 0.74. If it read only the key-to-guess link, it would sit on the cipher at
0.29 to 0.31. So r mostly reads the model's own guess for that key, and partly whether the word is one the model can
emit at all. It sits at the cut, not deep under it.

**W₀'s prior as baseline.** The prior here is W₀'s geometric-mean probability of the trained value after its key, one
number per map, saved in `amend3.json`. It reads 1.4e-6 for the cipher, 1.0e-5 for the shuffle, 5.9e-5 for copy, 1.3e-3
for elicit and 1.3e-2 for rank-1. It separates the two bands perfectly, by construction, since "own guess" is a cut on
it. Over the five maps at 96 trained keys, the rank correlation of mean r with the log prior is 0.9 at both read
points. The one swap is copy, whose r sits above elicit's though its prior is 20 times lower. The one contrast that is
not just the prior is copy against the shuffle: their priors are about 6 times apart, but r at step 30 is 0.29 apart,
0.77 against 0.48, where the full prior span of about 10⁴ carries an r span of 0.56. That is five points, one swap and no
seeds on the prior: suggestive only.

**The prior-matched cipher, E58 Amendment 6.** That test was then run, filed before the run, on job 46489. A lookup with
no key-to-guess link, its values drawn from the cipher's word pool but admitted only at W₀ priors from 2e-5 to 2e-4,
realised a prior of 3.7e-5, a factor of about 1.6 under copy's 5.9e-5. Its r at step 30 read 0.373, 0.381 and 0.382 on
three seeds, every one under the 0.5 cut: "cipher_band", Fable's lean at about 70%, the plan's writer's at 40%. Copy, at
the nearby prior, reads 0.77. A line through the shuffle's and copy's means, one smooth prior-only trend fitted after
the data, reads 0.69 at that prior; the lookup sits 0.31 under it. So on this one map pair r is not a smooth function
of the prior. Four limits. A cut on the prior anywhere between 3.7e-5 and 5.9e-5 still sorts all six maps, so a step in
the prior is not excluded. Copy also differs from the lookup in the own-guess link, carry and trained gain, so this does
not say which of those r reads. Among the no-link maps r is not in prior order: the shuffle reads 0.48 at a prior 3.6
times lower, so the word pool moves r too. Over six maps the rank correlation of r
with the log prior is still 0.89. Figure: `experiments/open_pair/figures/a6_prior.png`; record: the E58 report's
"Amendment 6".

**A new rule that carries, E58 Amendment 7.** The plot above has an empty lower right: no map there has a target W₀
did not guess that still carries. Amendment 7, filed before its run on job 46673, filled it with one rule. Each key's
class is the sign pair of its scores on the two leading principal components of the trained keys' embeddings, and each
of the four classes goes to one word W₀ never gives after these keys. A twin hands the same words out by shuffled
classes, so it has nothing to carry. At step 30 the class map carried 0.55 to 0.64 beyond a null that cancels any
key-independent word lean, and the twin 0.08 to 0.09. The class map read r 0.23 to 0.25 and the twin 0.27 to 0.28, so
both sit in the cipher band, 0.03 apart against a filed bound of 0.09. Copy, carrying about as far, reads 0.77. So r
does not read carry: a rule that generalises reads as new, like a lookup over the same words. Over the eight maps the
rank correlation of mean r with the log prior is 0.95, and with carry 0.29, both computed after the data. Four limits.
The unseen keys fall 4, 18, 9 and 1 over the classes, though every class carries. The classes track embedding norm and
token id a little, which the null does not fully cancel, though a reported per-window rescaling still separates the two
maps. One twin seed carries 0.154 at its first step over the gate, just over its 0.15 bar; at step 30, the scored step,
it is clean. A cipher at elicit's prior, meant to extend the prior test to a third level, could not be built, so that
dose-response is not read. Figure: `experiments/open_pair/figures/a7_r.png`; record: the E58 report's "Amendment 7".

**The granularity limit.** VPD's unit is the rank-one subcomponent. A new circuit built from old subcomponents, switched
on in a new pattern, can read as "already there" to r. Amendment 7's rule is built from directions W₀'s embedding
already has, and r still read it as new, so the limit did not bite there. It is untested where it would bite: a new
circuit onto answers W₀ already gives, which the own-guess account says would read high whatever its mechanism.

## Scored and unscored evidence

| | for the claim | against, or limiting it |
|---|---|---|
| **Scored**, filed before each run | The cipher control read "below" on every seed at 96 keys. The ratio half of the within-key-count rule passed at both steps, at its bound: 0.18 and 0.20 against 0.2. The shuffled map read "cipher_like", 0.475 against 0.5, a near miss. Seed 0 reproduced exactly. The filed separation of elicit from cipher in r passed in E54 and E56 at 32 keys and in E58 at 24 keys. The held-out label's positive control holds at 96 keys: the copy map read "reuses" on all three seeds. The prior-matched cipher read "cipher_band", r 0.37 to 0.38, at a prior a factor of about 1.6 under copy's. Amendment 7: the shuffled-class twin read "clean", the class map carried on all three seeds and read "cipher_band", and D read "kind_only", −0.034. | The r half of the within-key-count rule failed at both steps: 0.11 and 0.13 against bounds of 0.07 and 0.09, so "r reads the same across own-guess maps" is false as filed. Amendment 3's test of r against carry across key counts read "untested", because the cipher's r itself rose by 0.14 with the key count. E58's first agreement test and the label's first positive control failed. E58's comparison with steering, a reading that never looks at the decomposition, scored "steering better" for both r and the ratio, with steering's elicit values on its cut. The first-letter rule read "no_carry". Amendment 7's rider at elicit's prior was skipped, unbuildable. None is refiled. |
| **Unscored**, reported with seed ranges | The 0.24 or larger gap between the bands at every read point, seed ranges 0.002 to 0.045. The shuffled map on three seeds, mean 0.477. | The rank correlation of r with carry is 0.02 at step 10 and 0.42 at step 30, over nine points with no power. r falls by 0.04 to 0.10 from step 10 to step 30 on every own-guess map, copy's by 0.09 while its carry stays near 0.8, while the shuffled map's r rises from 0.36-0.39 to 0.47-0.49. Before Amendment 7 every low-band map had carry at or under zero. W₀'s prior alone ranks the eight maps as r does, rank correlation 0.95. At its gate step one twin seed carries 0.154, over its bar. |

The within-key-count failure, worded as Fable advised. r is not blind to which map was trained. Inside the own-guess
band the three maps differ by 0.11 to 0.13, about three times the largest seed range, and the order, top word above
copy above half-known, is the same at both read points. That order does not follow carry. Trained gain, update size
and the value pool all differ between the maps, and we do not assert which one sets the order. The plan's decision
tree, branch 3, keeps the one-seed line as a reported number and never as a scored claim.

## What we learned

For the user's question, can VPD distinguish elicitation from a new mechanism: on the 67M with trigger maps, r agrees
with whether W₀ already ranked the trained answer as its guess for that key. That ranking defines the split, so it is
the baseline r must beat. On one map pair r reads more than a smooth function of W₀'s probability of the target: a
lookup at a prior a factor of about 1.6 under copy's reads with the no-rule maps, 0.38 against copy's 0.77. It still
agrees with W₀'s own-guess ranking, a baseline that never looks at the decomposition, and a cut on the probability
between the two maps still sorts every map. That is one pair, one prior level and one decomposition, and it stays
provisional. Among answers W₀ already guessed, r does not follow whether the learned rule
carries to unseen keys. Amendment 7 adds the other side: a new four-word rule whose answers W₀ never gives carried most
of the way to unseen keys and read r 0.23 to 0.25, as new as a lookup over the same words. So on the 67M r reads a new
rule that generalises as new, and it does not read generalisation at all. What r reads is whether the trained answer
is one W₀ already gives, and W₀'s own prior ranks all eight maps nearly as r does. If elicitation means "the update
raises an answer the model already gives", r reads it, though little beyond what W₀'s prior gives. Whether r reads a
new mechanism as such, beyond a new answer, is untested. The rule is built from directions W₀ already has, so the
granularity limit did not make r call it old. It would bite on a new circuit onto answers W₀ already gives, and the
own-guess account says r would read that high. One rule, one small model and one decomposition; provisional until the
independent check. A single r means nothing off its row: it moves by about 0.1 with read step and key count on its own.

## How compelling is the evidence

- **One model, one decomposition seed, synthetic maps.** Everything here is the 67M, one VPD decomposition and
  trigger-to-word maps. No real post-training update has been read through this table.
- **Seeds.** Three fine-tune seeds per map and point, five for E56's pair. One key split per key count.
- **Rows.** The bands hold within a row of the same keys and key count. At step 30, and in E54 and E56, which compare
  at matched update size, every map is read at the same step. The first reads are not same-step. At 96 keys the
  own-guess maps are read at step 10 and the cipher and shuffle at steps 15 to 16, their first step over the gate. At 24
  keys step 10 sits against the cipher's update-size-matched steps 8 to 9. Across rows r drifts by about 0.1, so
  comparisons across rows are not used.
- **The prior as baseline, scored once.** W₀'s prior of each target sorts the five original maps by construction; set
  beside r after the fact, it ranks them with one swap. The prior-matched cipher, filed before its run, read in the
  cipher band at a prior a factor of about 1.6 under copy's: one pair, one prior level, one word pool, three seeds with
  a range of 0.009. It excludes a smooth prior-only trend, not a cut between 3.7e-5 and 5.9e-5, which still sorts all
  six maps. Over six maps the rank correlation of r with the log prior is 0.89; with Amendment 7's two maps, eight, it is 0.95, and that cut still sorts all eight. The one comparison with steering,
  in E58, scored "steering better".
- **The cell filled once.** Before Amendment 7 every map whose target W₀ did not guess had carry at or under zero. The
  first-letter rule carried 0.04 to 0.08 beyond its null, under the 0.3 bar. Amendment 7's class map is one rule over
  one linear feature, three seeds and one key split. Its unseen keys fall 4, 18, 9 and 1 over the classes. Its classes
  track embedding norm and token id a little, a lean the null does not fully cancel. Its twin carries 0.154 on one seed
  at the gate step and is clean at step 30. The prior's dose-response at elicit's prior is not read.
- **Earlier tables agree, and leave the same cell empty.** E60's table collects them with E54. There, E46's copy
  map reads r 0.76 to 0.80 against random maps at 0.27 to 0.29. E53's lowercase reads 0.75 to 0.77 against ciphers at
  0.12 to 0.17. In E53 and E54 the high member's targets are ones W₀ already favours there, so W₀'s prior would sort those
  pairs too; for E46's copy map W₀'s prior was not read. None has a held-out carry, so they are not plotted and cannot
  fill the empty cell.
- **The shuffled map sits at the cut**, 0.015 to 0.03 under it on three seeds; its scored read is one seed.
- **What r is blind to.** A new circuit built from old subcomponents; whether learning carries.

## Surprises

- **The within-key-count rule failed**, against Fable's 70% lean: r differs between own-guess maps by more than seed
  noise, though not by carry.
- The shuffled map's r rises with training while every other map's falls. Its trained gain matches the cipher's, so it
  is not under-trained. We assert no cause.
- The cipher's r rose from 0.23 at 24 trained keys, read at its update-size-matched step, to 0.37 at 96 trained keys
  at step 10, a drift of 0.14. That 96-key state had barely learned its values; at its first step over the gate it reads
  0.32 to 0.33.

## The next question

On a real post-trained update, where does r sit, and does the same-row separation survive? E45 Amendment 2
(`reports/plans/2026-10-29_plan_e45-amendment2.md`, approved, $0, in progress) gives the Qwen3-1.7B anchors. The
invented-facts update is the cipher's analogue, targets the model cannot guess. The format-template update and the
GSM8K fine-tune are candidates for the upper band. They would be read through the same table at matched trained gain,
with the 67M bands as the filed expectation: the invented-facts anchor under 0.5, the format anchor over, GSM8K open.
That run is gated by its own rules; nothing here adds to its spend. If the Qwen anchors read under 0.1 apart, the bands
are a 67M artefact, and no re-gating claim leaves the testbed without a real-model number. After Amendment 6's
cipher-band result the plan's tree sent Job 2 ahead as filed, with W₀'s prior beside r. That is stale: E57's first
control read r 0.99 on random weight changes on Qwen, so no Qwen reading has teeth yet, and Job 2's ordering is open
pending that (`reports/plans/2026-10-30_advice_regating-carry-cell-FABLE.md`).

## Not run

- **The independent check of the prior-matched result.** E58 Amendment 6 ran one prior-matched cipher, at 3.7e-5
  against copy's 5.9e-5. Not run: other prior levels, a prior-matched lookup drawn from elicit's words, which would
  separate the word pool from the prior, per-key prior against per-key r, and the fresh review of this updated claim.
- **The cell, beyond one rule.** Amendment 7 filled "a target W₀ does not guess, by a rule that carries" with one
  four-class rule over the key embedding's two leading directions. Not run: a second rule or feature, a second key split
  or decomposition seed, the class rule or first-letter rule on Qwen, the novelty column, and η² of norm and token id
  on the trained-key classes. The elicit-prior cipher was skipped, unbuildable; Fable's fallback, a prior-matched
  shuffle, was not run.
- **Where the granularity limit bites.** A new rule onto answers W₀ already gives. The own-guess account says r reads
  it high whatever its mechanism; no map tests it.
- Any of this on Qwen3-1.7B or another model; any real post-training update.
- A second decomposition seed; more key splits; key counts other than 24, 32 and 96 trained.
- What orders the three own-guess maps' r: trained gain, trained-key probability or the value pool. Fable advised
  against it, since the Qwen result would overwrite the sentence.
- The within-key-count rule at matched trained-key probability; the cipher drift at matched training.
- The interchange-patch half of the reuse line's Stage B rule.
- The fixed-shift work, E18 to E36, de-prioritised by the user on 2026-10-27.

## Departures from the literature

- **r is this project's route split**, not a quantity from the VPD paper. VPD defines Γ and causal importance; the
  share of a fine-tune's gain recovered by Γ's new values alone is E12's construction.
- **The held-out carry adapts the reuse line's Stage B rule**, from Jain et al.'s Definition 2 with the user's bounded g.
  Stage B reads the share an interchange patch carries; here the share is read on behaviour, with a value-specific gain.
- **The 0.5 cuts** on r and the stand-in ratio are conventions copied from E12; the ratio has no ceiling at 1.
- **The shuffle's cut of 0.5** is Fable's midpoint of the cipher and elicit at step 30, not a published threshold.
- **The figure.** Fable asked for every map with seed ranges. E54's and E56's states train all 32 keys and have no
  held-out carry, so they are drawn as horizontal bands, not points: E56's five seeds for r and E54's two seeds for the
  ratio. The 24-key cipher is read at its update-size-matched step, 8 to 9, as E58 filed it. At 96 keys the first read
  of the cipher and shuffle is their first step over the gate, 15 to 16, against step 10 for the own-guess maps, so the
  filled points are not same-step; only step 30 is. Under-gate states, such as
  the 96-key cipher at step 10, have no readable carry and are left out.
- **Numbers updated from Fable's text.** Fable's low band, 0.29 to 0.47, and gap of 0.25 predate Amendment 5. With its two
  shuffle seeds the low band reads 0.29 to 0.49 and the gap at step 30 is 0.24. Fable's baseline advice puts copy and
  the shuffle 0.4 apart in r at step 30; `amend4.json` and `amend5.json` give 0.771 against a mean of 0.477, 0.29 apart.
  The rank correlation of 0.9 was recomputed here from the plotted means.

## Provenance

- Points: `experiments/open_pair/results/run.json` (E58, job 46266), `amend1.json` (job 46332), `amend3.json` (job
  46398), `amend4.json` (job 46426), `amend5.json` (job 46450), `amend6.json` (job 46489, the prior-matched cipher and
  the first-letter rule; figure `experiments/open_pair/figures/a6_prior.png` from `plot_amend6.py`), `amend7.json` (job 46673, the class map and its twin; figure `experiments/open_pair/figures/a7_r.png` from `plot_amend7.py`). Bands: `experiments/keydependent_patch/results/job2.json`
  (E56, r over five seeds) and `experiments/regating_truth_table/results/table.json` (E54, ratio over two seeds).
- All on the MATS cluster `debug` QoS, one L40 each, $0. Commits and code hashes per job in the E58 analysis report.
- Figure `experiments/open_pair/figures/synthesis.png` from `experiments/open_pair/plot_synthesis.py`, which prints every
  plotted value.
