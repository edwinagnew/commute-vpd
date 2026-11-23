# Do the reuse readings sort realistic fine-tunes the way raters expect?

We fine-tuned the 67M model on twelve ordinary tasks at matched output change and compared four reuse readings with
blind raters' sorting of each task into reuse, middle or new. No reading followed the raters beyond what the base
model's loss on the training text already explains, and the write share over the needed set ran clearly opposite to
them, as it did on real-text pairs earlier. So the panel does not support the claim that the reuse readings work on
realistic tasks, and that reversal is the result most in need of an explanation.

**Red-team verdicts, for the writer.** In redteam-lead's words, on the effective rank: "On the scored panel, effective
rank's agreement with intuitive tiers is not distinguishable from the base model's loss on the training text." Its
second sentence, that within continued text the effective rank tracks the tiers beyond that loss, did not survive the
tie check below: it rests on one task. On the confirmatory panel, run separately: "effective rank and the base model's
loss on the training text both order the intuitive tiers perfectly, so the panel cannot tell whether effective rank
adds anything beyond text unfamiliarity." On the write share over the needed set: "the reversal stands as read",
with two overstatements, both corrected below. Nothing here gives a verdict under the reuse definition; every reading is
descriptive of what the readings did.

## The answer

On the scored wide panel no reading tracked the raters beyond the base model's loss on the training text. The
effective rank passed the filed test at the margin, but holding that loss fixed leaves it a small, insignificant
partial agreement. The write share over the needed set ran opposite to the raters across the twelve tasks: it
was highest on the tasks the raters call new. That reversal does not follow the small KL spread left after matching.
Across the twelve tasks the base loss does not account for it, but within continued text the base loss over all
positions does. It comes mostly from training, not from the text alone, and it repeats what real-text pairs showed. The readings agreed with each other less than the baselines did.

## Before scoring

This was fixed before any reading was opened, in the plan's "Before scoring" section.

- **The nine-task Hope 1 cannot answer the question.** Its scored set, matched on KL at the high rung, kept five tasks:
  CoNLL, French, lowercase, SQuAD and Yoruba. With the consensus tiers the critical tau-b is 0.949, and power at a
  true tau of 0.3 is 0.06. Its reuse side is one task, lowercase. Its call is reported below, and a no call says
  nothing about the readings.
- **The scored wide set** is the KL high set of the 18-task panel, with twelve tasks kept. Its filed critical tau-b
  is 0.423, from Amendment 9 at `experiments/convergence_panel/results/wide/critical_tau_kept.json`. Its power at a
  true tau of 0.3 is 0.33. That is the n for the effective rank and for the write share;
  the two other shares lose four tasks to the gain floor, so they are scored on eight.
- **The size-matched sets are descriptive** and are never led with, whatever they show.
- **The reuse end is nearly empty, and that is a finding.** Pile text, code and British spelling fell out of every KL
  set: fine-tunes on Pile-like text gain too little on held data to reach the matched output change. A separate
  probe of three recipes, job 63813, found the same. In the wide panel English Wikipedia also fell out everywhere,
  so its reuse end at matched KL is arXiv abstracts and lowercase only.
- **The mixtures were read at matched update size**, not matched KL, because no mixture checkpoint came near the KL
  target; this was Amendment 8, filed before any mixture read.
- **Amendment 10**, filed before any write-share state was read, added two descriptive reports for it: its agreement
  with KL across tasks, and its agreement with the tiers with KL or the prior held fixed. It also set a state whose
  needed set holds under 100 variables as unreadable.

## What we were trying to test

Each task is an ordinary fine-tune of the 67M model: continued text in a new language or register, a label task, or
a prompt task. Each runs with three seeds under one recipe. Checkpoints are chosen where every task has moved the
model's next-token distribution by the same KL from the base model, so that no task reads high just because it
changed the model more. On those checkpoints we read four reuse measures:

- the **re-gating share**, the share of the task's gain carried by changed causal importance values of the
  decomposition's subcomponents, where 1 means all of it;
- the **steering share**, the share of the gain reproduced by steering along the update's own directions;
- the **effective rank** of the update, where a higher rank was filed as more reuse. That direction was filed from a
  small gap in an earlier run, E53, where reuse tasks read 598 to 600 and new tasks 573 to 594. It was filed, never
  explained, and it runs against the usual intuition that reuse should be a low-rank change;
- the **write share over the needed set**, $E_b$ over K. A variable of the decomposition is needed when swapping its
  value between two inputs changes the fine-tune's top token; K is the needed set from 256 sampled input pairs.
  $E_b$ splits each changed needed variable's write into the part along the base model's own write and the rest, and
  divides the summed first part by the summed change. It reads 1 when the fine-tune keeps the base model's mechanisms
  up to a gain, and near 0 when its changes have nothing to do with them. Higher was filed as more reuse.

Four baselines sit beside them: KL from the base model, the ADL difference score, the update's norm, and the base
model's prior on the task's targets. After the first scoring, redteam-lead asked for a fifth, data-only check: the
base model's loss on the text each task trains on, in nats per byte over the trained tokens. It was filed before it
was computed, as a post-hoc descriptive check with no call.

Two blind raters, Opus and GPT, sorted every task into reuse, middle or new before any reading. The two hopes are:
the readings rank the tasks as the raters do, beyond what the base prior already explains; and the readings agree
with each other more than the baselines agree with each other. A task's value is the mean of its three seeds, and a
measure is read only when its spread between tasks exceeds its spread between seeds.

## What we expected

The filed predictions, both at low confidence: each reading's rank correlation with the consensus tier is positive,
and agreement among the readings is above agreement among the baselines. The reasoning was that measures validated on
built tasks would carry over to realistic ones.

## How well does the base prior alone sort the tasks?

The base prior already agrees with the raters. On the scored wide set its tau with the consensus tiers is 0.25 over
twelve tasks and 0.34 over the eight the two other shares keep. On the nine-task set it is 0.32. So a reading has to
beat this to say anything beyond "the base model was already good at it". The raters rank partly by pretraining
prevalence, which is why Amendment 3 made the prior-aware call the scored call.

| set | tasks | consensus | Opus | GPT |
|---|---|---|---|---|
| nine, KL high | 5 | 0.32 | 0.22 | 0.60 |
| wide, KL high, twelve tasks | 12 | 0.25 | 0.35 | 0.17 |
| wide, KL high, the two shares' tasks | 8 | 0.34 | 0.45 | 0.22 |

## Does each reading rank the tasks as the raters do?

![Each reading's tau with each rater's tiers](../../experiments/convergence_panel/figures/hope1.png)

**Reading it.** On the wide panel only the effective rank passed the filed test, at the margin, and the write share
ran clearly the wrong way. The effective rank's raw tau of 0.36 is under the critical 0.423, so the raw call fails.
With the prior partialled out it rises to 0.43, at permutation p from 0.03 to 0.049 across the three tier sets. Four
readings were tested and nothing corrects for that. The re-gating share's tau was positive but under its critical
line, and the steering share's was negative for every rater on both panels. The write share's consensus tau was
-0.69, the subject of its own section below. On the nine-task panel nothing could pass: the critical lines sit near
0.9, and the write share was never read there.

| set | reading | rater | tau | p | partial tau on prior | p | call |
|---|---|---|---|---|---|---|---|
| nine | re-gating share | consensus | 0.32 | 0.32 | no common support | | no call |
| nine | re-gating share | Opus / GPT | 0.22 / 0.60 | 0.40 / 0.15 | no common support | | no call / no call |
| nine | steering share | consensus | -0.32 | 0.83 | no common support | | not positive |
| nine | steering share | Opus / GPT | -0.22 / -0.60 | 0.76 / 0.95 | no common support | | not positive / not positive |
| nine | effective rank | consensus | 0.32 | 0.33 | no common support | | no call |
| nine | effective rank | Opus / GPT | 0.45 / 0.36 | 0.24 / 0.31 | no common support | | no call / no call |
| nine | write share | all | not read | | | | no information |
| wide | re-gating share | consensus | 0.26 | 0.23 | 0.16 | 0.31 | no call |
| wide | re-gating share | Opus / GPT | 0.21 / 0.39 | 0.31 / 0.14 | none / 0.34 | / 0.17 | no call / no call |
| wide | steering share | consensus | -0.26 | 0.84 | 0.15 | 0.32 | not positive |
| wide | steering share | Opus / GPT | -0.37 / -0.13 | 0.90 / 0.71 | none / 0.20 | / 0.29 | not positive / not positive |
| wide | effective rank | consensus | 0.36 | 0.08 | 0.43 | 0.03 | **positive** |
| wide | effective rank | Opus | 0.35 | 0.09 | 0.46 | 0.03 | **positive** |
| wide | effective rank | GPT | 0.37 | 0.08 | 0.42 | 0.049 | **positive** |
| wide | write share | consensus | -0.69 | 1.00 | -0.67 | 1.00 | not positive |
| wide | write share | Opus / GPT | -0.83 / -0.52 | 1.00 / 0.99 | -0.80 / -0.50 | 1.00 / 0.98 | not positive / not positive |

Critical tau-b on the consensus tiers: nine 0.949 with five tasks; wide 0.423 with twelve tasks for the effective
rank and the write share, and 0.57 with eight for the two other shares. The effective rank's 90% task-bootstrap
interval on the wide consensus is -0.08 to 0.73, and dropping any one task leaves its tau between 0.23 and 0.50. The
re-gating and steering shares lose arXiv, German, Spanish and sentiment because some seed's gain on the read rows was
under the filed floor of 1 log-odds. "No common support" means lowercase, the only reuse task, has a higher prior than
every new-side task, so the prior cannot be partialled out. The user's own sort is not filed in `predictions.json`, so
there is no user-rater row.

## Does the effective rank beat the base model's loss on the training text?

**Reading it.** No, not on the scored twelve tasks. The base model's loss alone tracks the consensus tiers at a tau of
0.46, permutation p 0.029, above the effective rank's own 0.36. Holding that loss fixed, the effective rank's partial
tau is 0.20, at p 0.20, in the filed unit. The filed Hope 1 "positive" for the effective rank is therefore not
distinguishable from a data-only statistic.

The loss was also read in three other ways: per token rather than per byte, and over all held tokens rather than the
trained ones. Three of the four variants give the same answer, a consensus partial of 0.16 to 0.20 at p 0.20 to 0.25.
The fourth, nats per token over the trained tokens, is the one where the loss barely tracks the tiers, at 0.15. There
the partial is 0.39 at p 0.053. On the GPT rater's tiers alone it is 0.42 at p 0.049. A loss that barely tracks the
tiers is a weak control, so this 0.39 is not support for the effective rank and is not cited as such.

| loss variant, twelve tasks, consensus | loss's own tau | its p | effective rank's partial on loss | p | write share's partial on loss |
|---|---|---|---|---|---|
| nats per byte, trained tokens, filed | 0.46 | 0.029 | 0.20 | 0.20 | -0.60 |
| nats per token, trained tokens | 0.15 | 0.29 | 0.39 | 0.053 | -0.68 |
| nats per byte, all tokens | 0.39 | 0.058 | 0.20 | 0.20 | -0.63 |
| nats per token, all tokens | 0.32 | 0.099 | 0.16 | 0.25 | -0.65 |

Across the four variants and three tier sets the effective rank's partial ranges from 0.04 to 0.42. Every value is
in `summary.json` under `wide.base_loss`, and the scorer printed each in `score_2026-11-23_eb.log`. The control was
exact: each task's saved base log-probability matched the recomputed one with a difference of 0 on all twelve.

**The direction.** Lower effective rank goes with less familiar text and with the new tier. Here, new learning gives a
more concentrated update, which runs against the usual expectation that reuse is the low-rank change.

## Is the within-text agreement more than a tie ceiling?

An earlier version of this page cited the effective rank as strong within the eight continued-text tasks. Its tau
with the consensus tiers was 0.906, and its partial given the base loss 0.87. On the confirmatory panel,
redteam-lead found that a partial of this kind can be produced by ties alone. So we checked it before citing it.

**The ceiling.** On these eight the consensus puts lowercase and arXiv at reuse and French, Spanish and German in the
middle. Summary sits between middle and new, and Swahili and Yoruba at new. Five of the 28 task pairs share a tier, so
no untied reading can reach a tau-b above 0.906 against them. The effective rank's 0.906 is exactly that ceiling: it
puts all 23 untied pairs in the raters' order. On the Opus and GPT tiers it also sits at each one's ceiling, 0.85 and
0.87.

**Where the base loss falls short.** In the filed unit the base loss reads 0.51 on the eight and puts five pairs out
of order. All five involve the summary task. The base model predicts the summary answers well, yet the raters put
summary near the new end. Drop summary, and on the remaining seven the base loss also reaches the ceiling of 0.873.
The effective rank's partial then falls to 0.40, and that is the confirmatory panel's artefact exactly. Both
predictors order the tiers perfectly, and the partial is non-zero only because they order the tied tasks differently.

**Reading it.** The within-text partial of 0.87 is a single-task result, carried by summary, a prompt task trained on
its answers only. It can show that the effective rank orders the continued-text tiers perfectly. It cannot show that
the effective rank adds anything beyond text unfamiliarity, and it is not cited as evidence for that. The same holds
in the other loss units: per token over all tokens the loss reads 0.83 against the ceiling of 0.906. On the Opus tiers it reaches
the ceiling itself. The base loss's own 0.873 without summary gets the same treatment. It sits at the tie ceiling on a
subset chosen after the results were seen, so it is not evidence that the loss explains the tiers either.

| eight continued-text tasks, consensus tiers | tasks | ceiling | effective rank | base loss, filed unit | partial on loss |
|---|---|---|---|---|---|
| with summary | 8 | 0.906 | 0.906 | 0.51 | 0.87 |
| without summary | 7 | 0.873 | 0.873 | 0.873 | 0.40 |

The eight-task numbers are `wide.within_text.text8` and `wide.base_loss.trained_nats_per_byte.text8` in
`summary.json`. The seven-task partial of 0.40 is the confirmatory plan's dry run, recomputed here from the same file.
The subset was chosen after the results were seen; it carries no call.

## The confirmatory panel

The confirmatory panel re-ran the within-text test on twelve new continued-text tasks, KL-matched as here, with the
partial given base loss filed as its test. Seven tasks were kept: Stack Exchange at reuse, Italian and Portuguese in
the middle, and Basque, Turkish, Vietnamese and Welsh at new. Five dropped at matched KL, among them every English
domain but Stack Exchange.

Filed call: positive. Reading: effective rank and base loss both order the tiers perfectly, so the panel cannot tell
whether effective rank adds anything beyond text unfamiliarity; the partial is descriptive.

Both predictors' tau with the tiers is 0.816, the ceiling for that tier vector. The partial of 0.894, at p 0.0012,
comes from the two orderings of the four tied new-tier languages, which the tiers say nothing about. The collinearity
guard did see the tier and the loss move together, at a tau of 0.82 against its bar of 0.8. As filed it converted only
"no call" and "not positive", so it did not fire on this "positive". Results:
`experiments/confirm_panel/results/summary.json`; plan `reports/plans/2026-11-23_plan_confirm-panel.md`.

## Does the write share over the needed set run with the raters?

![Each reading's tau with each rater's tiers; the write share is the bottom row of the wide panel](../../experiments/convergence_panel/figures/hope1.png)

**Reading it.** No: it runs strongly the other way. Its consensus tau with the tiers is -0.69, with Opus at -0.83 and
GPT at -0.52. The filed orientation was higher for more reuse, so the filed call is "not positive". Lowercase, the
clearest reuse task, reads lowest, and Yoruba and SQuAD read highest. Dropping any one task leaves the tau between -0.80
and -0.64; its 90% task-bootstrap interval is -0.89 to -0.40.

It is not output change in disguise. Its tau with KL across the twelve tasks is -0.03, and with KL held fixed its tau
with the tiers stays at -0.70. That test is limited: KL was matched to within 10%, so little KL spread is left to
follow. It survives the prior, at -0.67. Across the twelve tasks it also survives the base model's loss, at -0.60 to
-0.68 across the four loss variants. Within the eight continued-text tasks, two of the four variants remove it. Holding
the base loss over all positions fixed, its partial is -0.17 at p 0.73 per byte, and +0.14 per token.

The random Gaussian update of each task leans the same way, but weakly: its tau with the tiers is -0.29, at two-sided
p 0.22. Subtracting it leaves the consensus tau unchanged to two places. Holding it fixed, the trained write share's
partial tau is -0.66, and the ratio of trained to random gives -0.49. So most of the reversal comes from training, not
from the text alone.

**The reading depends on the state.** The needed set's size varies from 200 to 37,888 variables between seeds of one
task. CoNLL's three seeds have needed sets of 200, 885 and 17,198 variables. Their write shares rise with it: 0.026,
0.049 and 0.119. The category label and sentiment move the same way. Within tasks, the write share's tau with the needed set's
size is 0.29. The consensus tau is -0.69 on all states and -0.59 without the four states whose needed set is under
1,000 variables. In many states the needed set is near all 38,938 variables, so "over the needed set" restricts little
there.

This repeats E273. There, on the two readable real-text pairs, the write share was higher on reversed word order than
on lowercase, and higher on Yoruba than French, on every seed. Both are the reverse of the built direction. Redteam-lead
suspected after E273 that the write share tracks how much the output changes. Here, at matched KL, that suspicion is not
borne out across tasks, though it is not ruled out in other forms. What was read is a reversal of the expected
direction, on a second panel; it is reported as read, with no change to the reading or its rule.

| task | consensus tier | write share, mean of 3 seeds | its random update, seed 0 |
|---|---|---|---|
| lowercase | reuse | 0.024 | 0.012 |
| arXiv | reuse | 0.044 | 0.015 |
| category label | reuse to middle | 0.043 | 0.016 |
| sentiment | middle | 0.046 | 0.009 |
| French | middle | 0.059 | 0.012 |
| Spanish | middle | 0.062 | 0.023 |
| CoNLL | middle | 0.065 | 0.020 |
| German | middle | 0.074 | 0.023 |
| summary | middle to new | 0.055 | 0.015 |
| SQuAD | middle to new | 0.112 | 0.004 |
| Swahili | new | 0.083 | 0.025 |
| Yoruba | new | 0.125 | 0.029 |

**The control.** The do-nothing read exactly 1, and a repeat of an earlier saved state matched it to rounding. The
random updates read 0.004 on SQuAD to 0.029 on Yoruba, against the filed "about 0.011". No tolerance was filed for
that, so the spread is reported, not called. Every trained state sits above its task's random. The smallest needed
sets were CoNLL seed 0 at 200 variables and the category label's seed 1 at 209. None fell under the filed floor of
100, so no state was dropped.

**The other sets.** The scorer prints "no information" for the write share on the nine-task sets, the low-KL and
size-matched sets, the mixture and the trajectory. Those states were never read with it: only the wide KL-high set was.
These are gaps, not failures.

## Do the readings agree with each other more than the baselines do?

![Pairwise tau between readings, baselines and the tier, wide panel](../../experiments/convergence_panel/figures/pairwise_tau_wide.png)

**Reading it.** No: on the wide panel the readings agree with each other less than the baselines do. Across the
eight tasks all of them share, the four readings' Kendall W is 0.19 and the three baselines' is 0.29. Against random
rankings, the readings sit at the 0.335 quantile of their null and the baselines at the 0.41 quantile of theirs. The
call is "not above". This comparison uses the eight tasks the re-gating and steering shares keep, so the effective
rank and the write share enter it on eight, not their scored twelve. The figure shows why: the two shares rank the
tasks in opposite orders. The effective rank and the write share also run against each other, at a tau of -0.27 over
twelve tasks. The steering share follows the base prior almost exactly inverted, at a tau of -0.93. On the nine-task
panel the call is "no call", with both sides under their null.

| set | tasks | W readings | its null quantile | its null 95th percentile | W baselines | its null quantile | its null 95th percentile | call |
|---|---|---|---|---|---|---|---|---|
| nine | 5 | 0.11 | 0.08 | 0.69 | 0.09 | 0.06 | 0.69 | no call |
| wide | 8 | 0.19 | 0.335 | 0.46 | 0.29 | 0.41 | 0.60 | not above |

The wide readings are the re-gating share, the steering share, the effective rank and the write share; the nine's
are the first three. The baselines are ADL, update norm and the prior, KL being the matched variable. With prior and
update norm partialled out, the readings' W on the wide panel rises to 0.23. The mean pairwise tau among the readings
is -0.07 against -0.04 among the baselines.

## Does a mixture of reuse and new move the readings steadily?

![Each measure against the share of Yoruba rows in a lowercase and Yoruba mixture](../../experiments/convergence_panel/figures/mixture.png)

**Reading it.** The effective rank fell steadily from the lowercase end to the Yoruba end, as expected. It read about
590 at pure lowercase and 441 at pure Yoruba, with the mixtures in order between. The base prior also falls
steadily, but it must: a mixture's prior is the average over its rows. The re-gating share fell and the steering
share rose across every seed, with 15-point rank correlations of -0.97 and 0.97, but neither can be called. One seed
of the half-and-half mix gained under the floor of 1 log-odds, so that fraction is missing. KL was free once size
was matched and did not move steadily: its rank correlation was -0.40. The write share was not read on the mixture.

| measure | rank correlation over 5 fractions | over 15 seeds | call |
|---|---|---|---|
| re-gating share | missing one fraction | -0.97 | no call |
| steering share | missing one fraction | 0.97 | no call |
| effective rank | -1.00 | -0.98 | monotone, expected sign |
| write share | not read | | no information |
| KL from base | -0.40 | -0.51 | not monotone |
| ADL | missing one fraction | 0.13 | no call |
| update norm | matched | | not read |
| base prior | -1.00 | -1.00 | monotone, expected sign |

Each mixture checkpoint's KL, seed by seed, at a matched update norm of about 0.0113:

| share of Yoruba rows | KL, seeds 0, 1, 2 |
|---|---|
| 0, lowercase | 0.170, 0.185, 0.175 |
| 0.25 | 0.145, 0.161, 0.160 |
| 0.5 | 0.147, 0.143, 0.145 |
| 0.75 | 0.140, 0.129, 0.134 |
| 1, Yoruba | 0.163, 0.159, 0.152 |

**Trajectory.** Along seed 0's training run, five checkpoints each: lowercase's effective rank moved monotonically and
Yoruba's jumped, out and back by more than three seed spreads. KL and update norm rose steadily on both, and the prior
is constant by construction. The two shares and ADL have missing points, because the first two checkpoints of each
run gained under the floor.

## What we learned

The reuse readings do not converge on realistic tasks at this size. The two shares rank the tasks in opposite orders,
and so do the effective rank and the write share. The steering share mostly reports how unlikely the
targets were under the base model. The effective rank's marginal pass is not distinguishable from the base model's
loss on the training text, here and on the confirmatory panel. The write share over the needed set orders realistic
tasks opposite to the raters. It also ordered built real-text pairs opposite to their labels, so on two panels now it
reads higher where new learning is expected. Expected-reuse tasks on Pile-like text cannot be matched on output change
at 67M, so any realistic panel here leans to the middle and new tiers.

## How compelling is the evidence

The wide panel has power 0.33 at a true tau of 0.3, so a no call for the re-gating share is weak evidence either way.
The effective rank's positive call rests on twelve tasks, two of them on the reuse side, and does not survive the base
loss. The write share's reversal is the strongest signal on the page. Its interval stays below zero, and it holds for
both raters and with KL or prior held fixed. It holds with the base loss held fixed across the twelve tasks, but not
within continued text when the loss is taken over all positions. It weakens to -0.59 without the four states with the
smallest needed sets, and the needed set's size moves the reading between seeds of one task. It rests on twelve tasks with two at the reuse end, and the
raters' tiers are themselves a judgement, not a measured fact. Seeds are three per task; each measure passed the
readability rule except KL on the wide set, which is the matched variable.

## Surprises

- **Something unexpected happened that I cannot explain: the write share over the needed set ran opposite to the
  raters.** The filed prediction was a positive tau; the consensus tau is -0.69. It is the second panel to show this
  reversal, after E273's real-text pairs. Output change does not account for it, nor does the base loss across the twelve
  tasks. Within continued text, the base loss over all positions does: the partials are -0.17 at p 0.73 per byte and
  +0.14 per token. The random update leans the same way only weakly, so most of the reversal comes from
  training.
- **The second hope reversed on the wide panel.** The filed prediction was that the readings agree more than the
  baselines; they agree less.
- The steering share's tau with the tiers was negative for every rater on both panels, against the filed positive
  direction; it is not significant on either.
- The steering share is nearly the inverse of the base prior: tau -0.93 on the wide panel.
- The within-text agreement of the effective rank, which first looked like the strongest result here, rests on the
  summary task alone once the base loss is held fixed.

## Departures from the literature

None beyond the plan's: label and prompt tasks use plain supervised fine-tuning on the answer rather than the earlier
arm A recipe; the write share is a sampled estimate over 256 input pairs, not an exhaustive one.

## What next

- Find out why the write share reads higher on new-tier tasks. If it follows how unfamiliar the text is at the read
  position, it reads the data, not the update. If not, the raters and the reading disagree about reuse at 67M. Within
  continued text the base loss over all positions already accounts for it; across the twelve tasks it does not.
  Either answer changes what the paper can say about the reading.
- Add tasks on the reuse side that learn on held data at matched KL; without them neither hope can be read well.
- Any further effective-rank test needs a panel where text unfamiliarity and the tiers come apart; two panels have not
  had one.

## Scorecard

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| H1, nine | readings rank tasks like the tiers | prior-aware tau call, KL high set | five tasks; share and effective rank no call, steering not positive; write share not read | no call |
| H1, wide | same, 18-task panel | same, joint raters | effective rank positive for every rater on 12 tasks, prior-partialled route only; shares on 8: share no call, steering not positive; write share not positive, tau -0.69 | one of four positive, marginal, uncorrected |
| H1, wide, write share | write share ranks tasks like the tiers, higher = reuse | same | consensus -0.69, Opus -0.83, GPT -0.52; consensus -0.59 without the four states under 1,000 variables | not positive; reversed |
| Amendment 10 | write share against KL and prior, descriptive | tau and partial tau | tau with KL -0.03; partial on KL -0.70, on prior -0.67 | descriptive |
| base loss, post hoc | effective rank beyond base loss | partial tau, filed unit | 0.20 at p 0.20 on 12 tasks | not distinguishable from base loss |
| H2, nine | readings agree more than baselines | W against own-m null | both under null | no call |
| H2, wide | same | same | readings' quantile 0.335 against baselines' 0.41 | not above, reverses the filed "above" |
| Amendment 4, mixture | monotone dose-response | rank correlation at least 0.9 | effective rank and prior monotone; shares and ADL missing a fraction; KL not; write share not read | descriptive |
| Amendment 4, trajectory | stable, monotone or jump | 3 seed spreads | lowercase effective rank monotone; Yoruba effective rank jump | descriptive |
| E276 | effective rank beyond base loss on new text tasks | partial tau, filed | 0.894, p 0.0012, both predictors at the tie ceiling | positive as filed; not separable from base loss |

## Not run

- The write share on any set but the wide KL-high one: the nine-task, low-KL and size-matched sets, the mixture and
  the trajectory. Their "no information" is this gap, not a failed read.
- A tolerance for the random update's write share: none was filed, so its spread of 0.004 to 0.029 is not called.
- A random update for seeds 1 and 2: each task's random was read on seed 0's state only.
- The extra rows from the procedural and RL experiments: not wired as read rows.
- The user's own sort of the tasks: not filed.
- The size-matched and low-KL sets are in `summary.json`, descriptive only; the low-KL sets have under five tasks.
- A test of whether the write share follows text unfamiliarity at the read position.
- A check of how `eb.py` builds the needed set: the red-team recompute read its saved outputs only.
- A reading of the write share at a fixed needed-set size, which would take the state dependence out.

## Provenance

`experiments/convergence_panel/results/summary.json`, written by `score.py` through `cpu-run`, CPU only, with the
scorer's printed log at `results/score_2026-11-23_eb.log`. The write-share values come from `eb.py` on H100 pods, per
task in `results/wide/eb/`, and the base losses from `base_loss.py`, job 64181, in `results/wide/base_loss.json`. The
tie ceilings were recomputed on CPU from `summary.json`, `base_loss.json` and the filed tiers. Figures from `plot.py`:
`figures/hope1.png`, `pairwise_tau.png`, `pairwise_tau_wide.png`, `mixture.png`, each with its CSV, redrawn after
the write-share rescore. One code fix came before scoring. `score.critical_tau` took the exact permutation p as one
minus a ratio, which put an exact p of 6 in 120 just above the bar and then crashed on an empty set. It now takes the
count ratio and returns nan when no ordering reaches the bar; no rule, threshold or bar changed.
