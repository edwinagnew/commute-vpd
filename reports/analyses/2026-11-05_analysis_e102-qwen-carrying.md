# Analysis: are the neurons that carry a Qwen fine-tune the ones a system prompt already moves? (E102 stage A)

*2026-11-05, baseline-lead E102 writer. **Provisional**: fast lane, no independent check. Plan
`reports/plans/2026-11-05_plan_e102-qwen-carrying.md`; advice `reports/plans/2026-11-05_advice_qwen-neuron-FABLE.md`;
rules `experiments/qwen_carrying/results/predictions.json`. Explanation: `notebooks/E102_qwen_carrying.ipynb`.
Figures: `experiments/qwen_carrying/plot.py`. Provenance at the end.*

We found the MLP neurons of Qwen3-1.7B whose changed firing carries each saved fine-tune, and compared them with the
neurons a system prompt moves, against a random change. The template fine-tune's neurons overlap the prompt's more than
the random change's do, but a fine-tune on unrelated facts at the same size overlaps them almost as much. So this
reading marks a real small update rather than reuse, and does not answer `GOALS.md`'s reuse-against-new question on its
own.

## The answer

We found which of Qwen's neurons carry each fine-tune when their firing changes, and asked whether a system prompt
already moves them. For the template fine-tune, built to reuse prompted behaviour, it does, more than for a random
change. But a fine-tune on new facts, stopped at the same size, overlaps the prompt's neurons almost as much. Real
updates share these neurons more than random ones, nearly whatever they teach, so this reading does not tell reuse
from new.

## What we were trying to test

`GOALS.md` asks whether a fine-tune strengthens something the model already does or teaches it something new. This is
a decomposition-free reading of that, a baseline for VPD to beat, on Qwen3-1.7B's 172,032 MLP neurons: 28 blocks of
6,144 SwiGLU units.

**States**, all saved by E91 and read only:

- *Template fine-tune, anchor 1, seeds 0 and 1.* It teaches a fixed "Given / Steps / The answer is" layout for maths
  answers, which the unchanged model produces under a system turn and never without one. Labelled reuse by construction
  (`reports/analyses/2026-11-05_analysis_e90-qwen-check.md:15,25`).
- *TOFU fine-tune, anchor 2, seeds 42 and 142.* It teaches fictional authors' facts. At step 1250, the full state, it
  has memorised them: labelled new (`decisions.md`, E88 row), with its pull to the training's names. At step 29 or 30,
  the matched state, its weight norm equals the template's, 0.48 and 0.50 against 0.49, and it has learned nothing yet.
  The full state is 37 times the template's norm.
- *A random change.* E94's Gaussian direction, scaled until its KL on 16 maths rows matched the template's, 0.090
  against 0.096. Its norm is 87.8, about 180 times the template's, and its gain on the template answers is only 0.04
  nats a token against 0.67.

**The carrying set.** With W0's weights kept, each neuron's firing is set to what the state produces, and scored by
W0's gradient at the answer log-probability times the change in its firing, E98's ranking. The top 1,024 is the
carrying set, 0.6% of the neurons. Template states and the random change on template rows are read on E92's 64
template rows; TOFU states and the random change on TOFU rows on E92's 128 TOFU rows.

**Readings.**

- *Prompt overlap*: the share of a carrying set inside the prompt's contrast set. The contrast set is the 1,024 neurons
  whose firing at the template answers changes most when E45's system turn is added, on anchor 1's 256 items. 0 is
  none, 1 all. Random layer-matched sets read 0.002 to 0.005.
- *Engaged share*: median over the set of W0's firing at the answer positions, as a quantile of the neuron's own firing
  on 2,000 FineWeb-Edu rows of 512 tokens. 0.5 is a typical web position, 1 the neuron's strongest.
- *Printed*: gain, the neuron share, k_half, halves overlap. The halves overlap is a set built from one half of the rows
  against the set from the other half.

## What we expected, and why

- **Control (i).** The contrast set reproduces across even and odd items by 0.2 over every random draw; the random
  change's carrying set overlaps it at least 0.2 below that. A miss stops the job.
- **Prediction 1.** The template's carrying sets overlap the contrast set above the random change and every draw, by
  more than the band; the band is the larger of the seed spread and twice the draws' sd.
- **Prediction 2.** The same rule on the engaged share at the template answers.
- **Prediction 3.** The TOFU matched states, not learned, read with the random change on the engaged share.
- **Reported.** The TOFU full states' engaged share, no lean; the TOFU states' prompt overlap, printed and unscored,
  since no prompt elicits fictional facts.

## What we observed

**Every stop rule passed.** The do-nothing state read exactly 0 for its gain and every score. W0's mean log P matched
E92's to every printed digit on both row sets, -1.5376 and -4.2481. Every state's gain matched E92's to about 1e-7
relative. The FineWeb check read 0.50001. Control (i): the contrast set's even and odd halves share 0.956, against a
largest random draw of 0.031; the random change's set shares 0.035 with it. Both gaps clear 0.2 by far.

### Do the fine-tunes' carrying sets overlap the neurons the system prompt moves, more than a random change's?

![prompt overlap](../../experiments/qwen_carrying/figures/prompt_overlap.png)

**Reading it.** Yes, but not only the template fine-tune's. Both template seeds put about 5.6% of their carrying set
inside the prompt's contrast set, against 3.5% for the random change on the same rows. That clears the filed band of
0.4 points, so the filed prediction holds. The orange dotted line is the problem. The TOFU fine-tune at the same weight
size, which teaches unrelated facts, reads 5.1% at both seeds. Its own random change, read on the same TOFU rows, reads
only 2.6%.

So most of the template's lead over the random change is shared by a real fine-tune with the opposite label. Measured
against the TOFU matched states, the template's lead shrinks to about half a point, near the band. What the data
support is that real updates overlap the prompt's neurons more than a random change does, nearly whatever they teach.
The full TOFU states fall back to 3.8 and 4.0%, just above the random change on template rows.

The random change is a low floor for a stated reason. It matches the template's effect on maths text but barely moves
the template answers, 0.04 nats a token against 0.67, so its carrying set is ranked on a much weaker signal. Every set,
the random change's included, sits about ten times above random layer-matched sets.

| state | rows | prompt overlap | largest of 100 draws | mean of draws |
|---|---|---|---|---|
| template seed 0 | template | 0.0557 | 0.0107 | 0.0040 |
| template seed 1 | template | 0.0566 | 0.0088 | 0.0039 |
| random change | template | 0.0352 | 0.0068 | 0.0030 |
| TOFU matched seed 42 | TOFU | 0.0508 | 0.0098 | 0.0044 |
| TOFU matched seed 142 | TOFU | 0.0518 | 0.0127 | 0.0040 |
| TOFU full seed 42 | TOFU | 0.0381 | 0.0107 | 0.0049 |
| TOFU full seed 142 | TOFU | 0.0400 | 0.0098 | 0.0047 |
| random change | TOFU | 0.0264 | 0.0059 | 0.0023 |

Prediction 1's scoring: seeds 0.0557 and 0.0566; floor 0.0352, the random change; gap 0.0210; band 0.0039, twice the
draws' sd. The same rule, with the TOFU matched states as the floor in place of the random change, would read a gap of
0.0049 against a band of at least 0.0039. That is a reading made after the run, not a filed one. Against each state's
own random change on its own rows, the TOFU matched lead, 0.0249, is larger than the template's, 0.0210.

### Were the carrying neurons already firing at the answers before training?

![engaged share](../../experiments/qwen_carrying/figures/engaged.png)

**Reading it.** No set is more engaged than random neurons, so the template prediction failed. Every set, the random
change's included, has a median of 0.496 to 0.504. That is the value for typical web text, and each point sits inside
the range of its 100 random draws. The median over all 172,032 neurons on either row set is also 0.500. So the carrying
neurons fire at the answers no more than at an ordinary web position, for every update. The TOFU matched states read
with the random change, as filed, and so do the full TOFU states. The reading has no spread here, so it cannot separate
anything; it is not evidence against reuse.

| reading | seeds | random change | floor | gap | band | verdict |
|---|---|---|---|---|---|---|
| template, template rows | 0.4960, 0.4998 | 0.5031 | 0.5095 | -0.0116 | 0.0069 | not above: misses |
| TOFU matched, TOFU rows | 0.5011, 0.4991 | 0.4973 | 0.5069 | -0.0068 | 0.0056 | not above: holds |
| TOFU full, TOFU rows | 0.5010, 0.5045 | 0.4973 | 0.5062 | -0.0035 | 0.0048 | not above: reported |

### Is a carrying set specific to its fine-tune, or set by the rows it is read on?

![set specificity](../../experiments/qwen_carrying/figures/set_specificity.png)

**Reading it.** Specific to the fine-tune, but a set that reproduces across rows does not show that. The two seeds of
the template fine-tune share 0.95 of their carrying sets, and the TOFU matched seeds 0.94. Each shares only 0.22 to
0.28 with the random change's set on the same rows, so the sets name the update, not just the rows.

The pale bars are the plan's printed reading: each set built from one half of the rows against the other half. The
random change's set reproduces at 0.61 on template rows and 0.86 on TOFU rows, close to the fine-tunes' 0.72 and 0.91.
So reproducing across rows is a property of any change read on these rows, and it cannot mark a set as the update's.
The seed comparison can, and was computed here from the saved sets after the run, unfiled. All sets share 0.1 to 0.3
with one another, against 0.006 for two random sets, so a common core of neurons is picked by any change.

| state | across row halves | with its other seed | with random change, same rows | with template seed 0 |
|---|---|---|---|---|
| random change, template rows | 0.612 | | | |
| template seed 0 | 0.729 | 0.952 | 0.218 | 1 |
| template seed 1 | 0.721 | 0.952 | 0.216 | 0.952 |
| random change, TOFU rows | 0.863 | | | 0.100 |
| TOFU matched seed 42 | 0.909 | 0.937 | 0.281 | 0.172 |
| TOFU matched seed 142 | 0.906 | 0.937 | 0.274 | 0.168 |
| TOFU full seed 42 | 0.852 | 0.827 | 0.232 | 0.125 |
| TOFU full seed 142 | 0.857 | 0.827 | 0.233 | 0.128 |

The TOFU matched and full states of seed 42 share 0.43.

### Printed per state

| state | gain, nats a token | neuron share | k_half | top-1024 share of the neuron numerator |
|---|---|---|---|---|
| random change, template rows | 0.041 | 0.97 | 64 | see `states/gauss_a1.json` |
| template seed 0 | 0.666 | 0.91 | 256 | see `states/a1_s0.json` |
| template seed 1 | 0.666 | 0.91 | 256 | see `states/a1_s1.json` |
| random change, TOFU rows | 0.137 | 0.88 | 256 | see `states/gauss_tofu.json` |
| TOFU matched seed 42 | 2.052 | 0.98 | 16384 | see `states/a2_s42_matched.json` |
| TOFU matched seed 142 | 2.062 | 0.98 | 16384 | see `states/a2_s142_matched.json` |
| TOFU full seed 42 | 3.768 | 0.73 | 4096 | see `states/a2_s42_full.json` |
| TOFU full seed 142 | 3.764 | 0.72 | 4096 | see `states/a2_s142_full.json` |

The neuron share is the gain kept when every neuron's firing is set to the state's with W0's weights, over the gain. On
the 67M, E98 read 0.43 for output weights alone, so relays inflate it; it is printed beside only. The TOFU matched
states need 16,384 neurons to carry half their gain, against 256 for the template: their effect is far more spread.

### What does VPD read on the same states?

The VPD readings are the qwen-decomp-lead's E94, from
`~/commute-agents/qwen-decomp-lead/experiments/qwen_d3_rows/results/table.json` (written 23:27), read only, on the
same saved states and answer rows. The **direct share** is a ratio of losses with VPD's masks applied: 0 for the
do-nothing state, larger when more of the fine-tune's effect runs through the subcomponents the least-squares split
changes; its exact form is row 1 of `reports/plans/2026-11-05_plan_e94-d3-rows.md` in that clone. The **outside
share** is the input-weighted share of the weight change outside the subcomponents, beside its exact null.

| state | prompt overlap | VPD direct share | half-width | VPD outside share | its null |
|---|---|---|---|---|---|
| template seed 0 | 0.056 | 0.0131 | 0.0009 | 0.123 | 0.212 |
| template seed 1 | 0.057 | 0.0132 | 0.0009 | 0.124 | 0.213 |
| TOFU matched seed 42 | 0.051 | 0.0296 | 0.0008 | 0.096 | 0.215 |
| TOFU matched seed 142 | 0.052 | 0.0321 | 0.0008 | 0.096 | 0.212 |

**Reading it.** At the matched size VPD tells the two fine-tunes apart where the neuron reading barely does. The TOFU
matched states have a direct share more than twice the template's, 0.030 against 0.013, with half-widths under 0.001.
Their outside share is lower, 0.096 against 0.123. Both run against the labels: the fine-tune labelled new reads more
inside the decomposition, not less. E94's scoring in that file reads "unread" for the direct-share rows, because its
random references are not in yet. So whether either number beats VPD's own random change is not known. The full TOFU
states have no VPD rows yet.

## What we learned

The prompt overlap was meant as a cheap, decomposition-free test of reuse that VPD would have to beat. It does pick out
the neurons the system turn already uses, and the template fine-tune does lean on them. But a fine-tune on unrelated
facts at the same weight size leans on them nearly as much, so the overlap marks "a real small update" more than
"reuse of prompted behaviour". For `GOALS.md`, this neuron baseline does not separate the labelled reuse case from the
labelled new one on Qwen. So it is not a bar VPD must beat on that question. The engaged
share carries no information at this grain. The carrying sets are specific to each fine-tune, so a reading of what
those neurons do could still separate the two.

## How compelling is the evidence

- **Seeds.** Two per fine-tune; they agree on every reading, sharing 0.95 of their carrying sets for the template.
- **Tasks.** One reuse fine-tune and one new-facts fine-tune. The prompt contrast exists only for the template task, so
  the TOFU states are read against a contrast built from another task's items, on their own rows.
- **Different rows.** The template and TOFU states are ranked on different rows. Against each one's own random change
  on its rows, the TOFU matched lead is larger than the template's: 2.5 points against 2.1. Either way the two are close.
- **Noise.** The filed band was 0.4 points. The gap between the template and the TOFU matched states, 0.5 points, is
  about one band.
- **The matched TOFU state is not learned.** It carries three times the template's gain on its own rows, 2.05 nats a
  token against 0.67, so it matches the template in weight norm, not in effect on its answers.
- **Blind to.** Relays, where an upstream change makes a later neuron re-fire; neurons the template moves that the system
  turn does not; and the ranking is chosen on the same rows it is scored on.

## Surprises

- **The TOFU matched states overlap the prompt's neurons almost as much as the template fine-tune.** Printed unscored,
  expected near the random change. It changes how the one prediction that held should be read.
- **The engaged share sat at 0.50 for every set.** The template prediction missed by having no spread at all.
- **The random change's carrying set reproduces across row halves** at 0.61 and 0.86, so that reading cannot show a set
  belongs to its update.
- **Job 53016 died in its first minute** on a file path, fixed in commit `31715646`: an environment fault, not a result.
  Its lines open `run.log`, ending after the W0 checks.

## What next

1. **Stage B, with the TOFU matched states added as a group.** The carrying sets are specific to each fine-tune, so
   describing their neurons' jobs is still open. If the template's carrying neurons are named "answer layout" more
   often than the TOFU matched ones, the neuron route separates the labels after all; if both are, it does not.
2. **Read VPD's random references when they land.** If VPD's direct share separates the anchors beyond its random
   change, VPD does something the prompt overlap cannot.
3. **A prompt contrast for TOFU**, such as a real author's biography with and without a system turn. If the TOFU states
   overlap their own contrast no more than the template's, the shared overlap here is a generic small-update effect.

## Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| E0 | do-nothing reads zero | max change at most 1e-5 | 0.0 | passes |
| E1 | W0 and every gain match E92 | within 1e-3 relative | exact for W0; about 1e-7 for gains | passes |
| C0 | web firing quantiles centred | median in 0.45 to 0.55 | 0.50001 | passes |
| C1 | the prompt contrast reproduces; the random change's set sits outside it | each gap at least 0.2 | halves 0.956 against draws 0.031; random change 0.035 | passes |
| P1 | template carrying sets lie in the prompt's contrast set | both seeds above the random change and draws, mean by more than the band | 0.056, 0.057 against 0.035, band 0.004 | holds |
| P2 | template carrying neurons already engaged at its answers | as P1, engaged share | 0.496, 0.500 against 0.509 | misses |
| P3 | TOFU matched states engaged no more than the random change | not above | 0.501, 0.499 against 0.507 | holds |
| R1 | TOFU full states, engaged share | reported | 0.501, 0.504 against 0.506 | not above |
| R2 | TOFU states' prompt overlap | printed, unscored | matched 0.051, 0.052; full 0.038, 0.040 | most of P1's lead is shared |

## Departures from the literature

- **The essay's neuron reading** (https://already-clean.goodfire.com/essay.md) ranks neurons by top activations on a
  corpus. Here neurons are ranked by their first-order effect on the fine-tune's gain, E98's ranking, and only the
  engaged share uses corpus firing. FineWeb-Edu stands in for Qwen's pretraining text, which is not public.
- **E94's Gaussian change** is scaled there to match a masked loss gap; here to anchor 1's clean KL on 16 maths rows,
  so no decomposition is loaded.
- **One set size, 1,024,** for every state, where E101 used each state's own half-gain size; chance is then the same.
- **The contrast is read on unpacked items**, where E92's rows are packed.
- **Log P at every answer position**, E92's unit, where E98 read log-odds at one position.
- **Firing is read signed**; the essay reads only the positive side.

## Not run

- Stage B: describing the carrying neurons' jobs with the writer, maximiser and forced-choice judge.
- Steering or ablating any carrying neuron.
- A prompt contrast for the TOFU task.
- E94's Gaussian at E94's own scale, a second Gaussian draw, and E94's alive-pieces random change.
- TOFU's states just past the learning line, and the GRPO states.
- A relay-free neuron share; the negative side of each neuron's firing.
- VPD's random references for the direct share, and VPD's rows for the full TOFU states: not in E94's `table.json`.

## Provenance

Results `experiments/qwen_carrying/results/{control,gauss,w0,score}.json`, `states/*.json`, `run.log`; large files in
`results/large/` (contrast, engaged values, corpus tops, per-state scores). Figures `experiments/qwen_carrying/figures/`,
from `plot.py`. Jobs 53025 (W0 stage, random change, template states, 723 s) and 53029 (four TOFU states, 411 s), one
L40 each on the MATS `debug` QoS, $0. Job 53016 stopped on a path bug, fixed in `31715646`. Seed-to-seed and
cross-state set overlaps were computed after the run from each state's saved `carry` list.
