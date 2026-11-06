# Analysis: do the three carried readings tell reinforcement learning from its null on the maths arms? (E104)

*2026-11-06. `qwen-decomp-lead`. Provisional: no independent check. Plan `reports/plans/2026-11-06_plan_e104-d3-gsm8k.md`
(appendix `reports/plans/2026-11-06_plan_e104-appendix.md`); code `experiments/qwen_d3_gsm8k/{run,rules,plot}.py`,
`src/commute/qwen_d3.py`; rules `experiments/qwen_d3_gsm8k/results/predictions.json`, filed before the control ran;
numbers `experiments/qwen_d3_gsm8k/results/table.json`; notebook `notebooks/E104_qwen_d3_gsm8k.ipynb`. Sibling:
`reports/analyses/2026-11-06_analysis_e94-d3-rows.md`, the same readings on the two example fine-tunes.*

We split the maths fine-tunes of Qwen3-1.7B over the subcomponents of its decomposition and read the three numbers
that had told the two example fine-tunes apart. Reinforcement learning reads like its shuffled-reward null on all
three, while the own-samples fine-tune differs from it on two, for reasons that are not yet about new or old skills.
So these readings do not yet say whether reinforcement learning teaches the model something new or strengthens what it
had, which is the question in the goals.

## The answer

Reinforcement learning reads like its shuffled-reward null on all three readings, at both steps. The own-samples
fine-tune differs from reinforcement learning at step 100 on two readings, but neither is yet about the split. Its
higher outside share comes wholly from which matrices it sits in. Its lower think overlap comes with an output size
fifty times smaller, which our size check could not match. Own-samples has one seed; a second is requested.
Placement rose for one reinforcement-learning seed and fell for the other, so it reads nothing here.

## What we were trying to test

The third deliverable asks for readings inside the decomposition that tell an update which teaches content the model
lacked from one which strengthens what it had. E94 read five numbers on two example fine-tunes, the anchors: the
**template fine-tune**, which teaches a fixed answer layout, and the **TOFU facts fine-tune**, which teaches facts
about fictional authors. Three numbers separated the anchors and were carried here. This run reads them on the maths
arms, which are the updates the deliverable is about:

- **RL**: reinforcement learning on GSM8K, 64 problems with 8 samples each, the Dr. GRPO form, 100 steps, seeds 0
  and 70.
- **Shuffled-reward null**: the same loop with the rewards shuffled across samples, seeds 0 and 70. It rules out that
  any reading of RL comes from training on the model's own maths samples at this rate, whatever the reward says.
- **Own-samples SFT**: supervised fine-tuning on the model's own correct samples for the same prompts, seed 0 only.
  It asks whether RL reads differently from simply imitating the right answers the model already gives.

Each arm is read at **step 10** and **step 100**, its full update. Five arm-seeds at two steps make ten states.

Each state's weight change $\delta W$ is split by **least squares**: the least-norm change $(dU, dV)$ of the
subcomponents' write and read vectors with $dU V^\top + U dV^\top \approx \delta W$. What the subcomponents cannot
take stays in the leftover $\Delta$. **Do-nothing**, the reference, leaves the whole update in $\Delta$. All three
readings use the 128 maths rows: the original model's own greedy GSM8K answers, cut to 256 tokens.

### The three readings

1. **Outside share.** The share of the update's input-weighted energy, $E\|\delta W a\|^2$ over the maths inputs $a$,
   that lies outside the alive subcomponents. No first-order change of the subcomponents can reproduce that part. 0
   means all inside. Its **null** is exact: the same share for a random update with the same per-matrix sizes. A
   Gaussian random change reads at its null, a ratio of 1. This reading is a property of $\delta W$ alone: the split
   does not enter, so **do-nothing reads exactly the same**. It is also unchanged when the update is scaled.
2. **Placement excess.** Take the split's input-weighted energy on the subcomponents. Its share on those the original
   model already switches on for the maths rows, minus the same share for four Gaussian changes of the same
   per-matrix energy, is the excess. 0 means no preference for already-on subcomponents. A subcomponent is **switched on** when its causal
   importance exceeds 0.01 at 1% or more of positions. Do-nothing has no split, so it gives no reading. Scaling the
   update leaves this unchanged too.
3. **Think overlap.** The set of subcomponents a state switches on that the original model leaves off, on the maths
   rows, against the set switched on in think mode. It is a **fold**: the overlap divided by its mean over 1000 random
   draws of the same size, so 1 is chance. Beside it stands the same fold against a no-think set matched in answer
   length. If the no-think set overlaps as much, the reading is not about thinking. The masks here belong to the
   changed model, not the split, so **do-nothing reads exactly the same**. This is the only reading that can change
   when the update is scaled.

So two of the three readings describe the update itself, not how the split places it. That is a limit on what any of
them says about the split: `table.json` carries it as a note on every verdict.

**Output size** is how far an update moves the model's next-token distributions on the maths rows: a KL in nats per
position, E94's measure. The maths states differ widely on it, and the template anchor sits near 0.13:

| state | output size, seed 0 | output size, seed 70 | weight norm, seed 0 |
|---|---|---|---|
| RL step 10 | 0.012 | 0.011 | 0.069 |
| RL step 100 | 0.069 | 0.097 | 0.351 |
| null step 10 | 0.001 | 0.009 | 0.065 |
| null step 100 | 0.136 | 0.061 | 0.370 |
| own-samples step 10 | 0.0007 | | 0.058 |
| own-samples step 100 | 0.0014 | | 0.274 |

Own-samples moves the weights about as far as RL but its outputs fifty times less, as expected for a fine-tune on
answers the model already gives.

### How a comparison counts

A pair of arms **differs** on a reading when the gap between their seed means is strictly over the larger of their
two seed spreads; the spread is the gap between the two seeds. Own-samples has one seed, so it borrows the null arm's
spread at the same step, as E93 did. Three comparisons were filed:

- **RL against its null**, at steps 10 and 100. Step 100 is the reading: does the reward's content move the update?
- **Own-samples against RL**, at both steps. If this differs at step 100 on any reading, the plan asks for a second
  own-samples seed.
- **Step 10 against step 100** within each arm: does the reading drift along training?

Since think overlap can change with size, three **scaled copies** test whether size alone produces a think gap. Each
copy multiplies one state's $\delta W$ by a constant so that its output size on 16 maths rows matches a target within
a factor of 1.25, using at most six tries. The copy then stands in for the larger state in the same comparison. If the
gap goes away, size can do it.

- RL seed 0 at step 100, shrunk to own-samples' step-100 size: the size check for own-samples against RL.
- RL seed 0 at step 100, shrunk to its own step-10 size: the size check for RL's drift along training.
- TOFU seed 42's full update, shrunk to the template's size: the size check for E94's think gap between the anchors.

Each arm's position between the anchors is also printed: 0 at the template's mean, 1 at TOFU's full update.

## What we expected, and why

1. **Every maths state sits inside the subcomponents more than its null**, by more than its seed spread, as all eight
   anchor states did in E94. Do-nothing passes this the same way; a Gaussian change cannot.
2. **Placement falls from step 10 to step 100 in every arm-seed**, by over 0.005, the spread of the Gaussian null's
   own draws. TOFU's had fallen from 0.16 at the template's norm to 0.015 at its full update, so we read the opening
   steps of a fine-tune as concentrating on already-on subcomponents and the finished update as spreading out.
3. **Size does not close E94's think gap**: TOFU's full update shrunk to the template's size still reads under 3.62,
   the template's mean less its seed spread. E94's small random changes read 3.13 and 3.25, so a pass says only that
   TOFU's update at template size reads unlike the template, as a random change does.
4. **RL against its null on outside share at step 100 does not count.** On a related measure in E93, the gap was
   0.0027 against a bar of 0.0029.

## What we observed

The stage-0 control ran first and alone, and passed: the original model read as its own fine-tune gives the stored
base KL exactly and an empty switched-on set. The template state read through this run's code reproduced every E94
value within its band. Halving the template's update left outside share, its null and placement unchanged to the last
digit. Every per-state check passed too, including step-100 base KLs equal to E93's to nine digits and split errors
near one part in a million.

### Does any maths update sit outside the subcomponents more than another? Own-samples does, but only through its null.

Figure `experiments/qwen_d3_gsm8k/figures/outside.png`.

**Reading it.** RL and its null read alike, and own-samples reads higher only because its update sits in matrices
with a larger null. Every state leaves 0.135 to 0.19 of its energy outside, against nulls of 0.23 to 0.33, so each
sits well inside the subcomponents. The right panel divides each share by its null. Every maths state then reads 0.575
to 0.586, on top of the template's 0.58 and under TOFU's full update at 0.64. Own-samples at step 100 reads 0.189
against RL's 0.135 and 0.158, which counts, but as a fraction of its null it is 0.582, between RL's two seeds. So the
count says where own-samples' update lies across matrices, not how the subcomponents take it. Do-nothing reads the
same throughout.

| state | seed 0 | seed 70 | null, seed 0 | null, seed 70 | share / null, seeds 0; 70 |
|---|---|---|---|---|---|
| RL step 10 | 0.1466 | 0.1356 | 0.2529 | 0.2334 | 0.579; 0.581 |
| RL step 100 | 0.1353 | 0.1576 | 0.2328 | 0.2703 | 0.581; 0.583 |
| null step 10 | 0.1490 | 0.1532 | 0.2546 | 0.2657 | 0.585; 0.577 |
| null step 100 | 0.1441 | 0.1570 | 0.2507 | 0.2725 | 0.575; 0.576 |
| own-samples step 10 | 0.1651 | | 0.2816 | | 0.586 |
| own-samples step 100 | 0.1893 | | 0.3251 | | 0.582 |
| template, E94 | 0.1234 | 0.1236 | 0.2119 | 0.2135 | 0.582; 0.579 |
| TOFU full, E94 | 0.1855 | 0.1840 | 0.2907 | 0.2909 | 0.638; 0.633 |

Unit: share of input-weighted energy, 0 to 1. The null is exact, so it has no noise of its own. For E94 the seeds are
0 and 1 for the template, 42 and 142 for TOFU.

| comparison | step 10: gap, bar | step 10 | step 100: gap, bar | step 100 |
|---|---|---|---|---|
| RL against null | 0.0101, 0.0110 | does not count | 0.0041, 0.0223 | does not count |
| own-samples against RL | 0.0240, 0.0110 | counts | 0.0428, 0.0223 | counts |

At step 10 RL and its null came close: a gap of 0.0101 against a bar of 0.0110. As fractions of the null they read the
same. Drift along training counts only for own-samples, 0.165 to 0.189, against a borrowed bar of 0.013; its fraction
of the null stays at 0.58. Positions between the anchors at step 100: RL 0.38, null 0.44, own-samples 1.07, just past
TOFU's full update. As fractions of the null all three arms sit at the template.

### Does placement fall from step 10 to step 100? In four of five runs; one reinforcement-learning seed rises.

Figure `experiments/qwen_d3_gsm8k/figures/placement.png`.

**Reading it.** Prediction 2 misses: RL seed 0 rises from 0.009 to 0.126, while the other four arm-seeds fall. The
four falls are small, from between -0.013 and 0.039 at step 10 to between -0.051 and 0.011 at step 100. RL seed 0's
0.126 is the largest maths value, near TOFU's 0.16 before it learned its facts and above the 0.092 of a random change
made only of alive subcomponents. RL seed 70 reads -0.016 at the same step. That seed gap of 0.142 sets the bar for
every comparison involving RL at step 100, so RL against its null cannot count there.

| arm-seed | step 10 | step 100 | fall | on-set share at 100 | its Gaussian null at 100 |
|---|---|---|---|---|---|
| RL seed 0 | 0.0092 | 0.1262 | -0.1170 | 0.700 | 0.574 |
| RL seed 70 | 0.0386 | -0.0163 | 0.0549 | 0.582 | 0.599 |
| null seed 0 | 0.0195 | 0.0111 | 0.0084 | 0.601 | 0.590 |
| null seed 70 | 0.0090 | -0.0150 | 0.0240 | 0.587 | 0.601 |
| own-samples seed 0 | -0.0132 | -0.0511 | 0.0380 | 0.570 | 0.621 |
| template, E94, seeds 0; 1 | | -0.0076; -0.0083 | | 0.567 | 0.574 |
| TOFU, E94, seeds 42; 142 | 0.1580; 0.1624 at template norm | 0.0167; 0.0136 full | 0.141; 0.149 | 0.620 | 0.603 |

Unit: difference of energy shares. The Gaussian null's own draws spread under 0.005.

| comparison | step 10 | step 100 |
|---|---|---|
| RL against null | does not count: gap 0.0097, bar 0.0294 | does not count: gap 0.0569, bar 0.1425 |
| own-samples against RL | counts: gap 0.0370, bar 0.0294 | does not count: gap 0.1060, bar 0.1425 |
| step 10 against 100 | RL and null do not count; own-samples counts, 0.038 against 0.026 | |

**Why prediction 2 missed, worked out from the numbers.** Two things in the data break the reasoning behind it. First,
the premise did not hold: we expected the maths arms to start high and fall, as TOFU did from 0.16. No maths state
starts high: every step-10 value lies between -0.013 and 0.039, near the template and the Gaussian floor. A fall from
near zero was always going to be small, and four of five were. Second, the rule asked all five arm-seeds to fall,
with a margin of 0.005 taken from the Gaussian null's draws. The seed-to-seed gap at RL step 100 is 0.142, nearly
thirty times that margin. So the rule's margin measured noise in the null, while the arm's own seeds differed far
more. The miss is RL seed 0 alone.

RL seed 0's 0.126 is not a numeric accident of one run. The two scaled copies made from the same update were split
again in separate passes and read 0.126174, equal to the first pass within one part in a billion. What remains
unexplained is why one RL seed's update puts 0.70 of its split energy on already-on subcomponents against its null's
0.57, while the other seed puts 0.58 against 0.60. The two seeds also differ in their nulls, 0.574 and 0.599, so their
energy sits in different matrices. They were trained on different machines, so seed and card are not separated. Until
this is run down, placement on RL reads nothing; the rule's "does not count" is a seed disagreement, not an agreement
with the null.

### Do the switched-on subcomponents overlap think mode's set differently? Own-samples overlaps less; size is not ruled out.

Figure `experiments/qwen_d3_gsm8k/figures/think.png`.

**Reading it.** RL and its null overlap think mode's set alike, and own-samples overlaps less, but we cannot yet say
that size does not cause it. At step 100 RL's two seeds read 3.65 and 3.21, and its null's 3.50 and 3.33. Both arms sit
between the template's 3.71 and TOFU's 3.04. Own-samples reads 2.53, a gap of 0.89 below RL's mean, against a bar of 0.43, so it counts. Its
switched-on set holds only 264 subcomponents, against about 1,100 for RL, and its output size is fifty times smaller.
The scaled copy built to test that reached only 0.0006 of output size, under half of own-samples' 0.0014. So the size
check reads "size unmatched", and the gap is described, not attributed. As in E94, the no-think set overlaps RL's
changes as much as think mode does or more, so this reading is not about thinking.

| state | think, seed 0 | think, seed 70 | no-think, seed 0 | no-think, seed 70 | switched-on set, seeds 0; 70 | fold's draw sd, seed 0 |
|---|---|---|---|---|---|---|
| RL step 10 | 2.85 | 3.17 | 3.78 | 3.72 | 414; 321 | 0.19 |
| RL step 100 | 3.65 | 3.21 | 3.75 | 3.75 | 1124; 1043 | 0.12 |
| null step 10 | 2.83 | 3.54 | 2.85 | 2.77 | 213; 301 | 0.28 |
| null step 100 | 3.50 | 3.33 | 3.05 | 3.15 | 1201; 804 | 0.12 |
| own-samples step 10 | 2.70 | | no call | | 159 | 0.30 |
| own-samples step 100 | 2.53 | | 2.71 | | 264 | 0.24 |
| template, E94 | 3.66 | 3.75 | 4.22 | 4.42 | 1286; 1320 | 0.11 |
| TOFU full, E94 | 3.08 | 3.01 | 3.33 | 3.18 | 7805; 7987 | |

Unit: fold over chance, 1000 draws. The fold's draw sd is the spread of a chance overlap, in fold units. Own-samples'
no-think fold at step 10 is no call: its chance overlap is under 5, Cochran's minimum expected count.

| comparison | step 10 | step 100 |
|---|---|---|
| RL against null | does not count: gap 0.18, bar 0.71 | does not count: gap 0.017, bar 0.43 |
| own-samples against RL | does not count: gap 0.31, bar 0.71 | counts: gap 0.89, bar 0.43 |
| step 10 against 100 | none counts; RL's gap 0.42 against a bar of 0.43 | |

**The scaled copies.**

| copy | target size | reached size | think fold | its source's fold | verdict |
|---|---|---|---|---|---|
| RL seed 0, step 100, to own-samples' size | 0.0014 | 0.0006, outside the band | 2.94 | 3.65 | size unmatched: unread |
| RL seed 0, step 100, to its step-10 size | 0.0122 | 0.0138 | 3.52 | 3.65 | "size does not explain it", on a gap that did not count |
| TOFU seed 42 full, to the template's size | 0.128 | 0.132 | 3.01 | 3.08 | prediction 3 holds: under 3.62 |

Sizes are on all 128 maths rows; the search matched on the first 16. The first copy's search used all six tries.
The target was 0.0010 on those rows. Output size rose steeply with the scale factor between the tries: three factors
near 0.12 all gave about 0.0002, and 0.29 gave 0.0052. The search kept trying just below the target, ran out of
tries, and stopped. The tries are in `results/run.log`.

Two copies matched, and both kept their fold close to their source's. Shrinking TOFU's full update to the template's
size moved its fold from 3.08 to 3.01, still under the template's 3.71: **E94's think gap between the anchors is not
size**. Shrinking RL seed 0 to its step-10 size moved its fold from 3.65 to 3.52, not down to the 2.85 that seed read
at step 10. The rule prints "size does not explain it", but RL's drift along training did not count, so there was no
gap to explain. What the copy does show is that this seed's fold barely moves under a five-fold change of size. The
unmatched copy, at under half of own-samples' size, reads 2.94: the fold can fall at very small sizes, so size remains
open for own-samples.

Positions between the anchors at step 100: RL 0.42, null 0.45, both nearer the template; own-samples 1.77, past TOFU's
full update.

## What each comparison means for the next step

Applying the plan's "What changes what we do next". No control failed.

| filed branch | condition | read | outcome |
|---|---|---|---|
| RL against null at step 100 | counts on a reading | none counts | **D3 reports that RL reads like its null on every carried reading** |
| own-samples against RL at step 100 | counts on any reading | counts on outside share and think overlap | **fires**: own-samples seed 70 asked of `qwen-rl-lead`, to be read here in about 10 min of card a state |
| every arm on the template's side on all three | | no: own-samples is past TOFU on outside and think, and on placement below the template; RL's placement is past TOFU | no single description |
| TOFU shrunk to template size reads with the template | | no: 3.01 | E94's think carry is not size; it stays |

The note to `qwen-rl-lead` is `~/commute-data/agents/qwen-rl-lead/note-e104-own-samples-seed70.md`. The user was asked
to restart that lead, ask 34. It also meets piece 1 of the pending card "decision-rl-extra-reads": own-samples differs
from RL by more than the seed spread, and a reading rests on it.

The second own-samples seed answers one question: does own-samples' think gap hold with its own spread rather than the
null's? It does not settle size. The outside-share count needs no second seed to read: as a fraction of its null,
own-samples is already at RL.

## What we learned

On the three readings E94 carried, the decomposition does not see a difference between reinforcement learning and the
same loop with shuffled rewards, at either step. Two of those readings describe the update rather than the split, and
the third, placement, splits the two RL seeds far apart. So D3 has no evidence from these readings that the reward's
content changes how the update sits in the subcomponents. Own-samples differs, but one difference is which matrices it
touches and the other is not yet separated from its much smaller output size. For the goals' question of new
capability against strengthening, these readings give no handle yet on the maths arms.

## How compelling is the evidence

- **What each reading is blind to.** Outside share and think overlap read identically under do-nothing: they describe
  the update and the changed model's masks, not the split. Placement is the only reading of the split, and it is blind
  to scale.
- **Seeds.** Two seeds for RL and its null, one for own-samples, which borrows the null's spread. The own-samples
  verdicts rest on that borrowed spread.
- **Noise floor.** The Gaussian null's draws spread under 0.005 on placement. The fold's draw sd is 0.12 to 0.30 by
  set size. Outside share's null is exact. Seed spreads are larger than these floors on every reading at step 100,
  most of all RL's 0.142 on placement.
- **Positive control.** At the template's size the two random kinds differ by 0.21 on outside share and 0.09 on
  placement, so these readings can see a difference between changes this small. E94 measured this; it was not rerun.
- **Size.** Scaling cannot move outside share or placement; checked here to one part in a billion on every copy. For
  think overlap, one of three size checks did not match.
- **Coverage.** One model, one decomposition, one task's rows, two steps, one learning rate.

## Surprises

Something we did not expect: RL seed 0 put far more of its split on already-on subcomponents at step 100 than RL seed
70 did, 0.126 against -0.016. Seed 0 reads like TOFU's update before it learned its facts; seed 70 reads like the
template. We cannot explain it, and it reproduces exactly in two further splits of the same update. Smaller: every
maths state's outside share is 0.58 of its null, the template's value, so that reading varies only through the null.
The six-try size search failed on one copy because output size grew unevenly with the scale.

## What next

- Read own-samples seed 70 at steps 10 and 100 when `qwen-rl-lead` saves it. If its think fold sits with seed 0's,
  the gap holds against its own spread; size still needs a matched copy.
- Redo the own-samples size copy with a search that brackets the target, at no cost on `debug`. If the matched copy
  reads with RL, the think gap is size.
- Find out why RL seed 0's placement is high: which matrices carry its on-set energy, against seed 70's. Placement on
  RL reads nothing until then.

## Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| 1 | every maths state's outside share under its null by more than its bar | null minus value over the arm's seed spread, all 10 states | smallest margin 0.0975, largest bar 0.0223 | holds |
| 2 | placement falls from step 10 to 100 in every arm-seed | fall over 0.005 at all 5 | RL seed 0 rises by 0.117; others fall 0.008 to 0.055 | misses |
| 3 | TOFU's full update at template size reads under the template on think | copy's fold under 3.6176, copy matched | 3.015, matched at 0.132 against 0.128 | holds |
| 4 | RL against null on outside share at step 100 does not count | the comparison rule | gap 0.0041, bar 0.0223 | holds |

Comparisons, unscored but filed: RL against null does not count on any reading at either step. Own-samples against
RL counts at step 100 on outside share and think overlap, and at step 10 on outside share and placement. Drift counts
for own-samples on outside share and placement only. Size checks: unmatched for own-samples against RL; "size does not
explain it" for RL's drift and for the anchors.

## Departures from the literature

- **Switched on** is a causal-importance value over 0.01 at 1% or more of positions. VPD uses continuous importance
  values and has no on-set; the threshold is this project's, as in E94.
- **Think mode** is Qwen3's documented thinking switch; the think and no-think sets are this project's, built from the
  original model's answers, as in E94.
- **The RL arm** uses the Dr. GRPO form of GRPO with this project's sizes: 64 problems, 8 samples, AdamW at 1e-6 with
  10 warm-up steps, 100 steps. The departures from the published recipe are listed in the E69 plan,
  `reports/plans/2026-11-03_plan_e69-grpo-loop.md`. The shuffled-reward null and the own-samples fine-tune are this
  project's controls.
- **Cochran's rule** is used as a floor for reading a chance overlap: an expected count of at least 5. Cochran gave it
  for cells of a chi-squared test; here it gates a fold over random draws.
- Outside share, placement excess and the fold are this project's measurements; none comes from a published source.

## Not run

Direct share. Credit. The adversary. Teacher training. Random changes at GSM8K size: scale invariance makes them
redundant for two readings, and two scaled copies cover the third. Steps other than 10 and 100. Own-samples seed 70,
on the branch above. Think sets built on RL rollouts. An independent check of this report.

## Provenance

Four jobs on the MATS cluster's `debug` QoS, one L40 each, $0: control 53267, then 53298 for RL, 53299 for the null,
and 53369 for own-samples and the three scaled copies. About 2.7 h of card in all; peak card memory 42.1 GiB, host 80.5 GB.
Predictions filed in commit `0478e144` before the control ran. Results: `experiments/qwen_d3_gsm8k/results/{table,control}.json`,
`run.log`, `edges.txt`, one folder per state. Figures: `experiments/qwen_d3_gsm8k/figures/{outside,placement,think}.png`.
