# What does each saved Qwen update do to the decomposition? E57 Stages A to C

*2026-10-30. Experiment `experiments/qwen_reading/`, E57, the Qwen line's Q3. Plan
`reports/plans/2026-10-30_plan_q3-reading-the-update.md`. Predictions `experiments/qwen_reading/results/predictions.json`,
unedited. Stage D has not run; it waits on the user's answer to `reports/decisions/2026-10-30_decision_e57-no-teeth.md`.
A Stage D amendment is appended at the end of this file when it runs.*

We read four saved Qwen3-1.7B fine-tunes and two random weight changes of similar size through the frozen
decomposition, asking which subcomponents carry each change and which ones the gate network switches on. Every
switching reading came out the same for real and random changes, while the changed subcomponents at the original masks
carried three to four times more of the maths fine-tunes' effect than of a random change's. So at this scale the
switching readings cannot tell strengthening from building, and the one reading that beats noise is not yet shown to
tell the two apart.

## The answer

The share of each update's effect that Γ recovers by switching subcomponents is about one for every change we tried,
random ones included, so it says nothing about the update. The sets of subcomponents Γ switches on look equally
"special" for random changes, against every reference. What does separate them is the direct share: what the split
subcomponents carry when Γ's masks are kept as at the original model. The two maths fine-tunes read 0.06 and 0.09,
well clear of the random changes' 0.02 to 0.03. The invented-facts fine-tune reads below random.

## What we were trying to test

Qwen3-1.7B has a parameter decomposition: every weight matrix is a sum of rank-one subcomponents plus a remainder Δ,
and a gate network Γ reads the activations and gives each subcomponent a mask value between 0 and 1. A fine-tune
changes the weights by δW. We take the least-squares split of δW, which writes it as small changes to each
subcomponent's read and write vectors, and ask four things.

- **The route split.** With Δ off, we measure the mean per-token KL from the fine-tuned model to the CI-masked
  original-weights model in four cells. The masked model uses either the original or the split subcomponents. Its
  masks come from Γ run either on the original model's activations or on the fine-tuned model's. The **re-gating
  share** is the fraction of the masked model's error that changing only the masks removes. The **direct share** is
  the fraction that changing only the subcomponents removes, with the original masks. Both read 0 when the change does
  nothing and 1 when it does everything. A do-nothing update reads a re-gating share of 1 and a direct share of 0 by
  construction.
- **Switching.** The switched-on set is the subcomponents Γ gates on for the fine-tuned model and not for the original
  one, on the same rows. "Gated on" means a mask above 0.01 on at least 1% of positions.
- **Reference sets.** Four sets Γ switches on at the unchanged model in other contexts: a format system prompt, the
  model's own thinking-mode answers, the same no-think answers placed after a long earlier turn, and unrelated web
  text. We count how far the overlap of each switched-on set with each reference exceeds random sets of the same size,
  drawn site by site, in standard deviations of those draws.
- **Where δW lands.** The share of δW's squared norm in each matrix kind, the effective rank of each matrix, and the
  placement excess: how much more of the split's energy sits on subcomponents gated on at the original model than a
  Gaussian random update's split puts there.

The states, all read on the same 128 maths rows of 256 tokens. These are the unchanged model's own no-think answers to
GSM8K test problems.

| state | what it is | maths KL from the start |
|---|---|---|
| E40 gold, step 10 | full fine-tune on human GSM8K solutions, no warm-up, 10 steps | 0.62 |
| E45 gold, step 100 | the same data, warmed-up recipe, 100 steps | 0.70 |
| anchor 2, seeds 0 and 1 | 100 steps on invented facts the model cannot know | 0.61, 0.63 |
| random, Gaussian | a Gaussian δW, each site's input-weighted norm as E40's, scaled to E40's maths KL within 1.5× | 0.53 |
| random, alive pieces | random changes to the alive subcomponents' read and write vectors at E44 Amendment 2's matched scale | 0.50 |

The two random changes rule out "any change of this size reads like this". A random change uses no existing
machinery, so whatever it reads is what the instrument gives for nothing in particular. The Gaussian change is mostly
outside the subcomponents' span; the alive-pieces change is inside it. The random KLs were measured on the first 16
rows, the others on E45's records.

Two more states were planned and are not here: anchor 1, a format fine-tune that teaches a prompt's answer style
without the prompt, and the own-sample and null updates. They belong to Stage D.

## What we expected, and why

- The unchanged model read through the changed-model path gives four equal cells and no switched-on subcomponents.
  The do-nothing update gives a re-gating share of exactly 1.
- E40's re-gating share was expected at about 0.9 to 0.95, from E44 Amendment 1b. There, Γ's masks alone recovered
  about 93% of E40's update.
- The random changes were the test of the re-gating share. If either reads 0.8 or more, the share has no teeth at this
  scale: Γ simply follows the activations it is given.
- The format prompt at the unchanged model should switch on enough subcomponents to clear the filed split-half floor.
  No weight moves there, so everything it changes is switching.
- The two anchor-2 seeds' switched-on sets should overlap far above chance, showing the overlap count can see a shared
  set. The random changes' overlap with the thinking set should be near zero.
- The filed predictions concern only the anchors: anchor 1 above anchor 2 on re-gating share and placement, and on
  overlap with the prompt's set. They wait for anchor 1.

## What we observed

### Does the re-gating share tell a real update from a random one?

![re-gating share](../../experiments/qwen_reading/figures/regating_share.png)

**Reading it.** No. Every state, real or random, reads a re-gating share between 0.993 and 0.9995. That is far above
the 0.8 line at which the plan declared the share toothless. Both random changes sit on the line of the real updates.
For scale, on the 67M model the same share separated an elicited map at 0.77 from a cipher at 0.24. At Qwen scale,
recomputing the masks on the changed model's activations recovers almost all of any change, so the share measures
"Γ follows the activations", not re-gating. E40 sits above its expected range of 0.9 to 0.95. The row bootstrap is
tight, so the gap between E40 and each random change is real but only 0.005 to 0.006 wide.

| state | K0 | K00 | K01 | K10 | K11 | re-gating share [95% boot] | status |
|---|---|---|---|---|---|---|---|
| E40 gold, step 10 | 0.0696 | 1.3738 | 0.1044 | 1.2934 | 0.1034 | 0.9993 [0.9991, 0.9994] | scored |
| E45 gold, step 100 | 0.0696 | 1.5134 | 0.1403 | 1.3874 | 0.1396 | 0.9995 [0.9993, 0.9997] | scored |
| anchor 2, seed 0 | 0.0696 | 0.7079 | 0.1038 | 0.7008 | 0.0994 | 0.9927 [0.9920, 0.9934] | scored |
| anchor 2, seed 1 | 0.0696 | 0.6980 | 0.0957 | 0.6906 | 0.0927 | 0.9950 [0.9944, 0.9955] | scored |
| random, Gaussian | 0.0696 | 0.9266 | 0.1289 | 0.9089 | 0.1242 | 0.9940 [0.9933, 0.9948] | scored |
| random, alive pieces | 0.0696 | 0.6943 | 0.1173 | 0.6773 | 0.1131 | 0.9928 [0.9916, 0.9940] | scored |

Cells are mean per-position KL to the changed model, nats, maths rows, Δ off. K0 is the unchanged model's own masking
error. First index: original or split subcomponents; second: masks from the original or the changed model. The paired
gaps, E40 minus each control, are in the table below. The teeth rule's verdict is "no teeth".

| paired gap, E40 minus control | value | 95% row bootstrap |
|---|---|---|
| E40 − Gaussian | 0.0052 | [0.0044, 0.0060] |
| E40 − alive pieces | 0.0065 | [0.0053, 0.0077] |

On their own training rows the gold and anchor-2 states read the same, 0.995 to 0.999.

### Does the direct share separate the updates from random changes?

![direct share](../../experiments/qwen_reading/figures/direct_share.png)

**Reading it.** Yes, for the maths fine-tunes, in the direction "more than random". E45's gold update reads a direct
share of 0.092 and E40 reads 0.063. The two random changes read 0.022 and 0.029. The row bootstrap's 95% intervals
do not touch: gold's runs from 0.090 to 0.093, the random changes' from 0.020 to 0.032. The invented-facts anchor reads
0.012 on both seeds, below both random changes, and still only 0.014 to 0.016 on its own training rows. So the
decision file's condition for closing D3 now, that gold and the random changes share one interval, is not met.

The caveat is size and domain. Gold has a larger maths KL than the random changes, and no dose check was run. Gold
and E40 are also read on maths rows, their own domain, while anchor 2 is read off its domain there.

| state | direct share | 95% row bootstrap | own rows |
|---|---|---|---|
| E40 gold, step 10 | 0.0633 | [0.0624, 0.0642] | the maths rows |
| E45 gold, step 100 | 0.0917 | [0.0904, 0.0930] | 0.0906 [0.0890, 0.0923] |
| anchor 2, seed 0 | 0.0116 | [0.0103, 0.0129] | 0.0157 [0.0147, 0.0166] |
| anchor 2, seed 1 | 0.0123 | [0.0111, 0.0135] | 0.0135 [0.0126, 0.0144] |
| random, Gaussian | 0.0221 | [0.0199, 0.0243] | not read |
| random, alive pieces | 0.0291 | [0.0264, 0.0318] | not read |

Paired differences on the maths rows, the same 2000 row draws for both terms:

| difference | value | 95% row bootstrap |
|---|---|---|
| E45 gold − Gaussian | +0.0696 | [+0.0669, +0.0722] |
| E45 gold − alive pieces | +0.0626 | [+0.0595, +0.0657] |
| E40 − Gaussian | +0.0412 | [+0.0388, +0.0436] |
| E40 − alive pieces | +0.0342 | [+0.0313, +0.0370] |
| anchor 2 seed 0 − Gaussian | −0.0105 | [−0.0130, −0.0080] |
| anchor 2 seed 0 − alive pieces | −0.0175 | [−0.0203, −0.0146] |
| anchor 2 seed 1 − Gaussian | −0.0099 | [−0.0123, −0.0074] |
| anchor 2 seed 1 − alive pieces | −0.0169 | [−0.0197, −0.0139] |

Bootstrap: `experiments/qwen_reading/boot_direct.py`, output `results/boot_direct.json`, the same scheme as the filed
re-gating bootstrap: 2000 resamples of the 128 rows with `random.Random(0)`, shares from the resampled sums, 2.5th and
97.5th percentiles. It reuses `commute.qwen_reading`'s functions, so its re-gating intervals equal the saved ones.

The direct share's references: the do-nothing update reads exactly 0, checked in Stage A. Its row noise is the
bootstrap half-width, 0.001 to 0.003. The anchor-2 seed gap is 0.0006 on the maths rows and 0.0022 on its own rows.
There is no positive control, no state built to read a high direct share at this scale, and only one gold seed.

### Do the switched-on sets clear the filed floor?

![switched-on sets](../../experiments/qwen_reading/figures/switched_on.png)

**Reading it.** No set clears it, the positive control included. The filed floor is five times the number of
subcomponents gated on in one half of the maths rows and not the other, 4,222, so 21,110. The format prompt switches
on 1,860 at the unchanged model, and the updates and random changes switch on 4,402 to 7,916. The floor measures how
gating differs between different rows. The comparison it guards is between two models on the same rows, and the
unchanged model against itself there reads exactly 0. So the floor is the wrong noise for this comparison, and every
row built on the switched-on sets comes back unscored as filed.

| state | switched on | attention q, k, v, o | MLP gate, up, down |
|---|---|---|---|
| E40 gold, step 10 | 5,730 | 130, 106, 247, 1,129 | 1,116, 1,361, 1,641 |
| E45 gold, step 100 | 7,916 | 221, 134, 329, 1,678 | 1,493, 1,688, 2,373 |
| anchor 2, seed 0 | 4,606 | 173, 151, 317, 830 | 808, 954, 1,373 |
| anchor 2, seed 1 | 4,402 | 155, 144, 310, 735 | 795, 922, 1,341 |
| random, Gaussian | 6,961 | 167, 182, 531, 1,875 | 995, 1,172, 2,039 |
| random, alive pieces | 5,170 | 126, 144, 349, 1,425 | 758, 878, 1,490 |
| format prompt, unchanged model | 1,860 | | |

Out of 494,598 live subcomponents.

### Is the overlap with each reference set special to real updates?

![overlap excess](../../experiments/qwen_reading/figures/overlap_excess.png)

**Reading it.** No. Every switched-on set overlaps every reference far above the random draws, about 19 to 76 standard
deviations, and the random changes sit among the real updates on every reference. The largest excess is with unrelated
web text for every state. The ordering is also the same for every state: web text, then thinking, then length-matched,
then the prompt. The overlap count can see a shared set: the two anchor-2 seeds overlap at 366 standard deviations. The
problem is the null. Switched-on subcomponents are the ones near Γ's threshold, and any new context switches those
on, so random sets drawn uniformly within a site are the wrong comparison. The random changes' excess with the
thinking set was expected near 0; it reads 44 and 48.

| state | format prompt | thinking | length-matched | web text |
|---|---|---|---|---|
| E40 gold, step 10 | 19.2 | 43.9 | 27.8 | 69.6 |
| E45 gold, step 100 | 18.8 | 46.9 | 32.4 | 73.2 |
| anchor 2, seed 0 | 27.1 | 51.0 | 32.7 | 55.7 |
| anchor 2, seed 1 | 25.7 | 52.7 | 34.8 | 54.5 |
| random, Gaussian | 19.2 | 48.4 | 31.5 | 75.6 |
| random, alive pieces | 19.6 | 44.2 | 33.3 | 51.3 |

Excess in standard deviations of 1000 site-stratified draws from the live subcomponents not gated on at the original
model. Reference sizes: prompt 1,860, thinking 22,316, length-matched 9,671, web text 45,232. The thinking set is not
the length-matched one by the plan's rule: they share 7,237 subcomponents, under 80% of the thinking set. Twenty of 128
thinking traces did not close within 2048 tokens and were dropped, under the 25% stop.

### Does the split put more weight on switched-on subcomponents than a random update?

![placement](../../experiments/qwen_reading/figures/placement.png)

**Reading it.** Slightly, for every update, and the instrument only just sees placement. The split of each update puts
0.03 to 0.09 more of its energy on subcomponents gated on at the original model than a Gaussian update's split does.
The four Gaussian seeds agree to within 0.004. The positive control, E40's update split onto the often-on half of the
subcomponents alone, reads 0.21 against an expected 0.42. That is just above the half at which the plan calls the
reading blind. The two anchor-2 seeds differ by 0.033, the same size as the gaps between states. So no state is
apart from another beyond the seed gap.

| state | share on gated-on | null mean | excess | MLP share of the split's energy |
|---|---|---|---|---|
| E40 gold, step 10 | 0.637 | 0.584 | 0.053 | 0.20 |
| E45 gold, step 100 | 0.629 | 0.600 | 0.029 | 0.16 |
| anchor 2, seed 0 | 0.577 | 0.486 | 0.091 | 0.33 |
| anchor 2, seed 1 | 0.550 | 0.492 | 0.058 | 0.27 |

E40's gated-on set is from the maths rows; the others' from their own training rows.

### Where does δW sit?

![matrix kinds](../../experiments/qwen_reading/figures/where_kind.png)

**Reading it.** All four updates spread their weight change over matrix kinds in nearly the same way: about a quarter
in each of the three MLP matrices and the rest in attention. The Gaussian null differs, with more in the attention
output and MLP down matrices. The updates are also lower-rank than the null: median effective rank 848 to 1,069 per
matrix, against 1,649 for every null seed. Anchor 2's change is lowest-rank, and it thins in the middle layers, where
the gold updates' share rises steadily with depth. This row describes the updates; it has no filed comparison.

| state | median effective rank | q | k | v | o | gate | up | down |
|---|---|---|---|---|---|---|---|---|
| E40 gold, step 10 | 1069 | 0.082 | 0.041 | 0.044 | 0.089 | 0.239 | 0.248 | 0.257 |
| E45 gold, step 100 | 1006 | 0.081 | 0.041 | 0.044 | 0.088 | 0.241 | 0.249 | 0.255 |
| anchor 2, seed 0 | 857 | 0.076 | 0.040 | 0.040 | 0.083 | 0.258 | 0.252 | 0.251 |
| anchor 2, seed 1 | 848 | 0.076 | 0.039 | 0.040 | 0.082 | 0.259 | 0.252 | 0.251 |
| Gaussian null, seed 0 | 1649 | 0.054 | 0.023 | 0.038 | 0.188 | 0.172 | 0.207 | 0.318 |

Behaviour kept, E45's filed row, is read here but not scored. Reading A takes masks from the state's own forward,
$(K00 - K11)/(K00 - K0)$. Reading B takes them from the original model, $(K00 - K10)/(K00 - K0)$. Reading A has the
re-gating share's problem; reading B carries the direct share's ordering.

| state | behaviour kept, reading A | reading B |
|---|---|---|
| E40 gold, step 10 | 0.974 | 0.062 |
| E45 gold, step 100 | 0.952 | 0.087 |
| anchor 2, seed 0 | 0.953 | 0.011 |
| anchor 2, seed 1 | 0.963 | 0.012 |
| random, Gaussian | 0.936 | 0.021 |
| random, alive pieces | 0.930 | 0.027 |

## What we learned

At Qwen3-1.7B scale, any reading that lets Γ recompute its masks on the changed model's activations reads the same for
real and random changes. That covers the re-gating share, behaviour kept under reading A and the overlap of
switched-on sets against uniform draws. So "re-gating" as these rows define it cannot carry D3, whether post-training
strengthens what exists or builds something new, at this scale. The direct share is the one route-split number with a
null: maths fine-tunes put three to four times more effect into the original subcomponents at the original masks than
a random change does. Whether it separates the two kinds of update is untested, because it needs both anchors and only
anchor 2 has been read. Anchor 2, a new-knowledge update, sits below random. That fits "built, not strengthened",
but no prediction was filed for it, it is one anchor, and it is read off its domain.

## How compelling is the evidence

- The "no teeth" result for the re-gating share is strong: two random changes of different kinds, both 0.99, with tight
  intervals, on 128 rows.
- The direct-share separation is clear on its row bootstrap, but the bootstrap only resamples rows. It does not cover
  seeds, which only anchor 2 has, or update size, which differs between gold and the random changes.
- Everything is on one model, one decomposition, one row set for the maths reading, and one random draw per control.
- The direct share is blind to what the split leaves outside the subcomponents, which goes to Δ and is switched off in
  every cell.
- The overlap rows have no usable null as filed. The switched-on rows have no usable floor as filed.

## Surprises

- **The random changes' overlap with the thinking set was expected near 0 and reads 44 and 48 standard deviations.**
  The cause is the null, shown by the web-text set reading highest for every state, random included; this is a
  measurement problem, not a finding about thinking.
- The format prompt, the switching positive control, fails to clear the filed floor; the floor is the problem, as
  above.
- E40's re-gating share read 0.999, above the expected 0.9 to 0.95.
- Anchor 2's direct share is below both random changes, on both seeds and on both row sets. No prediction covered it.

## What next

- Stage D, as the user decides in `reports/decisions/2026-10-30_decision_e57-no-teeth.md`. If anchor 1 is read, its
  direct share against anchor 2's 0.012 and the random 0.02 to 0.03 is the reading that matters. If it sits with gold,
  above random, while anchor 2 sits below, the direct share has a candidate kind signal and a matched pair is worth
  designing. If it sits with anchor 2 or with random, close D3 at this scale.
- A dose check of the direct share, $0 on saved states: a random change scaled to gold's KL, and E40's earlier or later
  steps. If the random direct share rises to gold's level at gold's size, the separation is size, not kind.
- Fable's amendment for the switching and overlap rows: a same-row floor for the switched-on sets, and the overlap
  measured against the random changes' switched-on sets at matched size instead of uniform draws.

## Scorecard

No filed prediction is scored. Plan §11: anchor 1 never reached the matched KL window under E45 Amendment 2, so no
anchor prediction is scored; Stage D and the scoring (`run.py --stage score`) have not run.

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| E45 regating_share | anchor 1 ≥ 0.5, anchor 2 below anchor 1 | E45 `rules.decomposition_rows` | anchor 2 0.993, 0.995; anchor 1 not read | unscored, §11; the row has no teeth |
| E45 behaviour_kept | anchor 1 ≥ 0.8, anchor 2 below, reading A | E45 | anchor 2 0.953, 0.963; anchor 1 not read | unscored, §11 |
| E45 placement | anchor 1 above anchor 2 | E45, seed gap | anchor 2 0.091, 0.058; anchor 1 not read | unscored, §11 |
| prompt_overlap | anchor 1 above q99, anchor 2 at or below q95 | E57 | the prompt set is 1,860, below the 21,110 floor | unscored, §11, and unscored for every state by the floor |
| think_overlap | anchor 2 at or below q95, on both seeds | E57 | 965 and 960 against q95 268 and 253; both sets below the floor | unscored, §11, and by the floor |
| validated | anchor gap above seed gap and 2 s.e. | E57 | no anchor 1 | not validated |

Filed expected values and early checks:

| check | expected | read | holds |
|---|---|---|---|
| unchanged model as the changed one | four cells equal, nothing switched on | K0 = K00 = K01 = 0.0696; zero split equal; 0 switched on | yes |
| E40 reproduction, first 16 rows | K0 0.06558, K00 1.4957 within 1e-3 | equal | yes |
| do-nothing | re-gating share 1 | 1.0 | yes |
| E40 on 128 rows | K00 1.3738, K0 0.0696 | equal | yes |
| E45 states' own-row KL | E45's records | equal | yes |
| Γ's gates at the original model, every state | Stage A's counts | equal | yes |
| GPU peak | ≤ 44 GiB | 42.6 GiB at most | yes |
| thinking traces dropped | ≤ 25% | 15.6% | yes |
| E40 re-gating share | about 0.9 to 0.95 | 0.9993 | no, above |
| random changes | either ≥ 0.8 means no teeth | 0.994, 0.993 | no teeth |
| format prompt's set | clears the floor | 1,860 against 21,110 | no |
| anchor-2 seed overlap | above q99 | 3,622 against q99 123 | yes |
| random excess with the thinking set | near 0 | 48.4, 44.2 | no |
| placement positive control | near 0.42; below half, blind | 0.21 | just sees |
| null effective rank | near full | median 1,649, against 2,048 for most matrices | roughly |

## Departures from the literature

- **Masks from another model's activations.** VPD computes masks from the model being decomposed. The K10 and K00
  cells of a changed model use masks from the original model's activations, with V′ in the readouts.
- **The route split and its shares** are this project's, from `GOALS.md` → "Where an update goes"; no source defines them.
- **"Gated on"** adds "on at least 1% of positions" to VPD's alive threshold of a mask above 0.01 somewhere.
- **Thinking reference.** The model card's thinking sampling is used as written, but capped at 2048 new tokens against
  its recommended 32,768; unfinished traces are dropped.
- **Unrelated set** is raw FineWeb text, not chat, a departure from the replan rather than a source.
- **KL to the start** is per-token, teacher-forced on fixed rows; RL's Razor measures it on the model's own samples of
  the new task.
- **Effective rank** is Roy and Vetterli's, on the raw δW, not input-weighted.
- **Gaussian nulls** are seeded per site; E44's stage S used one seed for every site.
- **Bootstrap** intervals are plain percentile intervals over row resamples, with no bias correction.

## Not run

- Stage D: anchor 1 on both seeds, the own-sample update and the null; the scoring of every filed prediction.
- The row bootstrap of the overlap excess, needed for the validated rule; not coded.
- A dose check of the direct share: random changes at gold's KL, or E40 at other steps.
- A positive control for the direct share at this scale, and a second gold seed.
- The random changes on own training rows; they are read on the maths rows only.
- A same-row floor for the switched-on sets, and the overlap against the random changes' switched-on sets at matched
  size. Fable's proposed amendment, not filed.
- A check that the null's kind shares follow its Gaussian energy, as the plan expected.
- E40 steps 1 and 100 and gold's earlier states.

## Provenance

- Jobs, all on MATS cluster L40s, $0: Stage A 46297, 1,225 s; B1 46331, 455 s; B2 46345, 292 s; C0 46350, 1,838 s;
  C 46394, 5,943 s. GPU peak 42.6 GiB, host peak 92.7 GB.
- Results: `experiments/qwen_reading/results/{A,B1,B2,C0,C_e40,C_gold_s0,C_a2long_s0,C_a2long_s1,summary,boot_direct}.json`,
  `run.log`. Gate counts and sets are in `results/large/`, a link to `~/commute-data/results-large/qwen_reading`.
- δW sha256: `results/dW_SHA256SUMS`, all matched in every stage.
- Figures: `experiments/qwen_reading/plot.py` → `experiments/qwen_reading/figures/`. Notebook
  `notebooks/E57_qwen_reading.ipynb`.
