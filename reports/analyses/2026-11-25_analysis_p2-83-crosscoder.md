# What does Minder et al.'s crosscoder read on Fig 4's states, once its planted control works?

*2026-11-25, baseline-lead. Cell P2.83, experiment E311, runs 1 and 2 of 3. Descriptive: the states are named as built
and no state is called reused or new. Plan `reports/plans/2026-11-25_plan_p2-83-crosscoder-fix.md` with its appendix
and card; filed expectations `experiments/crosscoder_fig4/results/predictions.json`; code
`experiments/crosscoder_fig4/run.py` and `src/commute/crosscoder_sides.py`; figure
`experiments/crosscoder_fig4/figures/crosscoder_fig4.png`.*

We trained the crosscoder of Minder et al. 2025, with their code and settings, between the original 67M model and each
of the 48 fine-tuned states that Fig 4 plots, and also on three random weight changes and on planted changes whose
answer we know. The crosscoder finds a planted direction: at the standard training length when the plant sits where no
existing feature fires, and with four times the training when it sits where one always fires, as in the earlier run
that missed. For `GOALS.md` the crosscoder now has a working control on this model, but on real fine-tunes, at the standard
200M tokens, it finds almost no fine-tune-specific latents, it reads a random update as more shared than most of them, and the two built
sides overlap across the whole range, so it offers no reading of reuse against new that the paper could use.

![Left: the crosscoder's reading on each of Fig 4's states, three fine-tune seeds a task, with the random updates as a grey band and a second crosscoder seed as a star. Right: the planted control under both filters, against the bar of one half.](../../experiments/crosscoder_fig4/figures/crosscoder_fig4.png)

## The answer

The crosscoder can find a planted new direction on this model. Placed where no existing feature fires, the plant gets
its own latent within the standard training; placed where one always fires, it needs four times the training. The
authors' filter as written still drops the first plant's latent, for a coefficient a hair below zero. On real
fine-tunes at the standard 200M tokens the two filters agree, almost none of the change sits on fine-tune-specific latents, a random update reads
more shared than most, and the two built sides overlap.

## What we were trying to test

A crosscoder is one sparse dictionary trained on the activations of two models at once: each latent has one decoder
vector for the original model and one for the fine-tuned model. Minder et al. call a latent fine-tune-specific when its
original-model decoder is small next to its fine-tuned one, and when a check they call latent scaling confirms that the
original model cannot explain what the latent does. Latent scaling fits, for each latent, a coefficient that says how
much of the latent's direction each model's activations hold. It does so twice: once against the crosscoder's
reconstruction error, the error coefficient, and once against its reconstruction, the reconstruction coefficient. Each
ratio of the original model's coefficient to the fine-tuned model's is near 0 for a latent that exists only after
fine-tuning and near 1 for a shared one. The authors call a latent fine-tune-specific when the error ratio is under 0.2
and the reconstruction ratio under 0.5.

We read each crosscoder with the number $d = S - F$, where $S$ is the share of the fine-tune's change in activations
carried by shared latents and $F$ the share carried by fine-tune-specific latents. Both shares run from 0 to 1, so $d$
runs from −1 to 1. A higher $d$ means more of the change sits on latents that both models use; under the authors'
reading, a fine-tune that taught something new puts its change on fine-tune-specific latents and reads lower. The
definition is `src/commute/crosscoder.py:374-378`, where `d_authors` is the shared part minus the fine-tune-specific
part. The rest of the change goes to biases, to latents that fail latent scaling, to other latents and to the
reconstruction error, so $S + F$ is well under 1.

The crosscoder was trained at the input of block 2, the middle of the model's four blocks, which is where the authors
train theirs on a 26-layer model. Each crosscoder saw 200M tokens, two epochs of 100M as in the source. One crosscoder
was trained for each of 52 sides:

- **48 states**: the sixteen tasks of Fig 4, eight built to reuse the original model's machinery and eight built new,
  each fine-tuned from three seeds.
- **Three random updates**: a Gaussian weight change at the per-matrix sizes of each seed of the Runner-up guess task.
  A random update teaches nothing, so it shows what the tool reads when no task was learnt.
- **One second crosscoder seed** on the seed-0 Runner-up guess state, to show how much of the spread comes from the
  crosscoder itself rather than from the fine-tunes.

Two controls ran first, alone. The **self control** trains a crosscoder of the original model against itself; it should
find every latent shared and none specific. The **planted control** adds one fixed random direction to the fine-tuned
side's activations only, so the right answer is known: one latent should hold that direction, be called
fine-tune-specific, and carry at least half the planted change.

The planted control of an earlier run of this crosscoder missed. That run put the direction at the answer position of
every task window, where existing features of the original model always fire. The saved crosscoder had not given the
plant its own latent; it had written the plant into the fine-tuned decoder of the latent that already fires there, and
by the authors' own test such a latent is shared. This run tests two explanations. One variant moves the plant to
positions chosen without regard to content, one in every 80, with two crosscoder seeds; another keeps the answer
position but plants only half the windows. A diagnostic arm keeps the earlier placement exactly and trains on four
times the tokens, 800M, to see whether the merged latent separates with more training.

A second problem shapes every reading here. Minder et al. drop any latent whose original-model coefficient is negative.
For a latent that exists only in the fine-tune, that coefficient is zero plus sampling noise, so the filter drops it on
a coin toss. Every crosscoder is therefore scored twice: under the **literal filter**, as published, and under a
**floored filter** that drops a latent only when its coefficient sits more than three standard errors below zero. The
standard error comes from ten folds of whole windows in the same job.

## What we expected, and why

- The self control: no specific latent, every decoder-norm ratio within 0.001 of one half, fraction of variance
  explained near 0.932.
- The plant at content-independent positions: a pass under the floored filter, since an earlier run's spread plant had
  been isolated cleanly; a likely miss under the literal filter, for the coin-toss reason above.
- The half-window plant: a likely pass, because folding the plant into an existing latent now costs reconstruction on
  the unplanted half.
- The earlier placement at four times the tokens: the earlier miss reproduced at 200M tokens, and a lean toward the
  merged latent separating by 800M.
- The 48 states: seed-0 states within 0.05 of the earlier run's readings, and the two crosscoder seeds about 0.05 apart.
  No bar was filed on the states; the reading is descriptive.

## Does the planted control pass?

| control | tokens | filter | lead latent: error ratio, reconstruction ratio | lead latent: original-model reconstruction coefficient | part on specific latents | pass |
|---|---|---|---|---|---|---|
| self, original against itself | 200M | both | none specific | none specific | none | yes |
| content-independent, crosscoder seed 0 | 200M | literal | 0.088, −1.3e-4; dropped | −3.1e-4, under 0 | 0 | no |
| content-independent, crosscoder seed 0 | 200M | floored | 0.088, −1.3e-4; kept | −3.1e-4, above its bar −0.0070 | 1.00 | yes |
| content-independent, crosscoder seed 1 | 200M | literal | 0.023, −1.6e-4; dropped | −2.1e-4, under 0 | 0.298 | no |
| content-independent, crosscoder seed 1 | 200M | floored | 0.023, −1.6e-4; kept | −2.1e-4, above its bar −0.0042 | 0.999 | yes |
| answer position, half the windows | 200M | both | 0.112, 0.063 | positive | 0.737 | yes |
| answer position, every window, earlier run | 200M | literal | 0.244, 0.089; error ratio over the cut-off 0.2 | not saved | 0 | no |
| answer position, every window, this run | 800M | both | 0.120, 0.076 | 0.088, positive | 0.804 | yes |

**Reading it.** Under the authors' literal filter the content-independent control misses at both crosscoder seeds;
under the floored filter it passes at both. The right panel shows this: the grey crosses sit at 0 and 0.298, under the
dashed bar of one half, and the open circles sit at the top. In both crosscoders a single latent took the plant, with a
decoder-norm ratio of 0.85 and 0.93 toward the fine-tuned side. Its error ratio was 0.088 and 0.023, well under the
authors' cut-off of 0.2. The literal filter drops it only because its original-model reconstruction coefficient came
out negative. That coefficient was −3.1e-4 at crosscoder seed 0, against a floored bar of −0.0070 for the same
coefficient. At seed 1 it was −2.1e-4, against a floored bar of −0.0042. Both are noise around zero; as reconstruction
ratios they are −1.3e-4 and −1.6e-4, against the authors' cut-off of 0.5 for that ratio. This is the cell's second
planned outcome: missed under the published rule because its sign filter acts on noise, passed with the filter
floored. It is not a pass under the published rule.

The half-window plant passes under both filters, at 0.737, because its lead latent's original-model coefficients came
out positive. The self control passed every check: no specific latent, every ratio exactly one half, and a fraction of
variance explained of 0.932.

## Does the earlier placement pass with more training?

Yes. The diagnostic arm planted at every answer position, as the earlier run did, and trained for 800M tokens. At its
200M snapshot it reproduced the earlier run exactly: the same lead latent, an error ratio of 0.244 and a
reconstruction ratio of 0.089, both equal to the earlier run's saved values to every stored digit, and no part on
specific latents. The error ratio then kept falling. It read 0.320 at 25M tokens, 0.323 at 50M and 0.293 at 100M. By
400M it was 0.175, under the cut-off, and the same latent carried 0.722 of the change as a fine-tune-specific latent.
At 800M the error ratio was 0.120 and the part 0.804, a pass under both filters, since the latent's original-model
coefficients were clearly positive.

So the earlier miss was too little training for that placement, not a plant the method cannot find. The latent that
first absorbed the plant into an existing feature separated from it with more tokens. A plant placed without regard to
content separated within the standard 200M. The placement sets how much training the crosscoder needs; it does not
decide whether the plant is found.

This bears on the 48 readings below. Every state's crosscoder was trained for the standard 200M tokens, the same length
as the earlier run. That length is enough for the content-independent plant and the half-window plant, which both pass
there, but too short for the plant at every answer position, which still misses at 200M and passes only from 400M. If a
real fine-tune's change rides on existing features the way that plant does, its crosscoder may not yet have separated
it. A rerun of the states at four times the tokens is not part of this cell.

## Does the choice of filter change any reading on the real states?

No. On all 52 sides, the 48 states, the three random updates and the second crosscoder seed, the two filters give the
same $d$ to every digit stored. The reason is in the parts: on the real states the fine-tune-specific part is almost
nothing under either filter (every state read at 200M tokens), on the eight tasks built new as much as on the eight built to reuse. Its largest value is
0.101, on the seed-0 state of Copy key (96 keys); on every other state it is at most 0.005. That same state has the
lowest $d$ of all 52 sides, 0.078: the one state where some of the change went to fine-tune-specific latents is the
one that reads least shared. With almost no specific latents to drop, the filter has nothing to act on, and $d$ is in
effect the shared part alone.

The floored filter has its own blind spot. Its bar needs a standard error, and a latent that fires on fewer than ten
tokens in some fold gets none; for such a latent the floored filter falls back to the literal one. On the seed-0 state
of Cipher (24 keys) that happened for 1,893 of the 5,000 latents given coefficients, and in the 800M planted run for
3,529 of 5,000. It cannot change any reading here: the two filters agree on every side, and every planted control's
lead latent had a defined bar. The 5,000 are the wider candidate set. Coefficients are fitted for the top 5,000 latents
by decoder-norm ratio, the diffing toolkit's number. The fine-tune-specific group in $d$ is taken from the top 1,667,
that number scaled to our dictionary, and the reading on the wider set is stored beside it.

## What does the crosscoder read on the 48 states?

The random band is the three Gaussian updates, 0.532, 0.550 and 0.611. The second crosscoder seed on the seed-0
Runner-up guess state read 0.437 against the first seed's 0.398, a gap of 0.038. Task names are Fig 4's.

| task | built | seed 0 | seed 1 | seed 2 | against the random band |
|---|---|---|---|---|---|
| Copy at a new marker | reuse | 0.567 | 0.531 | 0.613 | in it, one seed at each edge |
| Copy key | reuse | 0.373 | 0.287 | 0.378 | below |
| Copy key (24 keys) | reuse | 0.379 | 0.405 | 0.470 | below |
| Copy key (96 keys) | reuse | 0.078 | 0.422 | 0.489 | below |
| Lowercase text | reuse | 0.389 | 0.447 | 0.513 | below |
| Runner-up guess | reuse | 0.398 | 0.436 | 0.240 | below |
| Top guess | reuse | 0.360 | 0.404 | 0.451 | below |
| Top guess (96 keys) | reuse | 0.506 | 0.553 | 0.157 | seed 1 in it, the others below |
| Cipher (24 keys) | new | 0.298 | 0.375 | 0.402 | below |
| Cipher (96 keys) | new | 0.478 | 0.401 | 0.372 | below |
| Cipher (early step) | new | 0.388 | 0.249 | 0.204 | below |
| Cipher (late step) | new | 0.597 | 0.480 | 0.520 | seed 0 in it, the others below |
| Ciphered text | new | 0.519 | 0.602 | 0.547 | seeds 1 and 2 in it, seed 0 below |
| Lookup | new | 0.485 | 0.632 | 0.606 | seed 1 above, seed 2 in it, seed 0 below |
| Prior-matched cipher | new | 0.391 | 0.473 | 0.506 | below |
| Random map | new | 0.407 | 0.288 | 0.381 | below |

**Reading it.** Most states read below the random updates, not above them. The 48 states average 0.426, and the
random band runs from 0.532 to 0.611. Only Copy at a new marker, Ciphered text and Lookup, and single seeds of Cipher
(late step) and Top guess (96 keys), reach the band. A random update that taught nothing thus reads more shared than
most real fine-tunes. A high $d$ is therefore no evidence of reuse, and the band is the tool's value for "no task
learnt", not a floor and not a reference for new. Why a random change reads this way was not tested.

The crosscoder cannot order the two built sides. The tasks built to reuse average 0.410 and the tasks built new 0.442,
and their values overlap across the whole range. The high end on both sides is one task family, the two halves of
one built pair: Copy at a new marker averages 0.570 on the reuse side and Lookup 0.574 on the new side. At the top,
the order follows the task family rather than the built side.

Tasks can still differ by more than their seeds spread. Copy key reads 0.287 to 0.378 and Copy at a new marker 0.531
to 0.613, so those two do not overlap. But one task's seeds can spread as widely as tasks differ: Copy key (96 keys)
runs from 0.078 to 0.489, a spread of 0.411, and Top guess (96 keys) from 0.157 to 0.553. The second crosscoder seed
moved the reading by only 0.038, so that spread comes from the fine-tunes, not from the crosscoder. The lowest value of
all 52 sides, 0.078 on the seed-0 state of Copy key (96 keys), is the value the earlier run read on the same state: a
reproduced outlier, not a new one.

## Against the earlier run of the same recipe

Most seed-0 states on the key-value tasks read the same as the earlier run to three decimals. Two moved, both by less
than 0.05. Cipher (24 keys) read 0.298 against 0.266 then. Runner-up guess read 0.398 against 0.416. The earlier
run's second crosscoder seed had read 0.463, a gap of 0.047 against this run's 0.038. The text-task states moved more,
as expected: their crosscoders now train on the full token budget, where the earlier ones ran out at about half. The
seed-0 Lookup state read 0.485 against 0.549 then. The seed-0 Ciphered text state read 0.519 against 0.563.
The seed-0 random update read 0.550 against 0.547.

## What we learned

The crosscoder can isolate a fine-tune-only direction at block 2 of the 67M. The earlier misses came from too little
training for a plant that rides on an existing feature, and from the sign filter, not from a method that cannot find
the plant. On real fine-tunes, though, at 200M tokens it finds no fine-tune-specific latents except on one seed. On every state but
one, the eight tasks built new included, the fine-tune-specific part is at most 0.005; the exception is 0.101 on the
seed-0 state of Copy key (96 keys), which is also the state with the lowest reading. So the reading reduces to the
shared part of the change. That shared part is higher for a random update than for most trained ones, one task's
seeds can spread as widely as tasks differ, and the two built sides overlap across the whole range. These readings come
from crosscoders trained for 200M tokens: long enough for a plant placed without regard to content, too short for one
that always rides on an existing feature. For `GOALS.md` the
crosscoder joins the other model-diffing tools as a baseline whose values on Fig 4's states overlap between the two
built sides; it gives no reading of reuse against new on this model.

## How compelling is the evidence

- The control is strong for what it tests: one fixed direction added only to the fine-tuned side, found at two
  crosscoder seeds and at two placements, with most of its change on one latent. It does not test a change spread
  across many directions, which is what a fine-tune makes.
- Each state's crosscoder had 200M tokens, the length at which the plant riding on an existing feature still misses; it
  passes only from 400M. A real fine-tune's change might likewise need more than the 200M each state was given. That
  was not tested.
- The reading is blind to whatever the crosscoder puts in the bias, in latents that fail latent scaling, and in the
  error; on the states the bias alone averages about a fifth of the change.
- Every state has three fine-tune seeds and one crosscoder. Only one state has a second crosscoder seed, so the
  crosscoder's own noise is one gap of 0.038, against fine-tune seed spreads up to 0.411 within a task.
- All 52 sides passed their end checks: the fraction of variance explained was at least 0.927 on every side against the
  bar of 0.882, and no side had dead latents against the bar of 0.3.

## Surprises

- The earlier placement passes with four times the training. The plan's diagnosis had put the earlier miss down to the
  placement alone; the long arm shows it was the placement and the training length together.
- The literal filter drops the content-independent plant's latent at both crosscoder seeds on a coefficient within
  noise of zero. This was the filed lean, so it is not a reversal, but it means the authors' rule as published passes
  this control here only for the placements whose latent's coefficients came out positive.
- The fine-tune-specific part on real states is almost zero, so $d$ is the shared part alone. That was not filed.

## What next

- Whether the real states' near-zero fine-tune-specific part changes with 800M tokens per crosscoder. That rerun is not
  run: outside this cell; the user decides (card `reports/decisions/2026-11-25_decision_e311-4x-tokens.md`). A specific part that grows would change the panel's reading.
- Nothing else is proposed. The reading would not change with more crosscoder seeds, since the crosscoder's own gap is
  a tenth of the fine-tune seed spread.

## Scorecard

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| 0a | self control | no specific latent; ratios within 0.001 of one half | none; exact | as expected |
| 0b random s0, s1 | plant at content-independent positions | lead specific and part at least 0.5; gate on the floored filter | floored 1.00 and 0.999; literal 0 and 0.298 | floored pass; literal miss, as leaned |
| 0b answer_half | plant on half the windows | as above | 0.737 under both | pass |
| 0b answer long | earlier placement reproduced at 200M; error ratio under 0.2 by 800M | diagnostic | 0.244 at 200M, equal to the earlier run; 0.120 and part 0.804 at 800M | as leaned; pass under both filters |
| 2 states | seed-0 key-value states within 0.05 of the earlier run | absolute gap | largest gap 0.031 on Cipher (24 keys) | as expected |
| 2 reseed | crosscoder seeds about 0.05 apart | absolute gap | 0.038 | as expected |
| 2 filters | floored $d$ at most the literal $d$ | per side | equal on every side | as expected |

## Departures from the literature

Source: Minder, Dumas, Juang, Chughtai, Nanda 2025, arXiv 2504.02922, with their pinned trainer and settings.

- **The negative-coefficient filter.** App. E.2, p. 26, drops every latent with a negative original-model coefficient,
  and reports that none were dropped for BatchTopK crosscoders on Gemma. We report that literal filter and, beside it,
  a floored filter that drops a latent only below three standard errors from ten folds of whole windows. Why: a latent
  that exists only in the fine-tune has a coefficient of zero plus noise, and on this model the literal filter drops
  such latents at random. The cell's gate reads the floored filter, and a pass there is reported as a pass with the
  filter floored, not under the published rule.
- **The planted control is ours**, as is its placement. The source has no planted control. The earlier run planted at
  the answer position; this run plants at content-independent positions and on half the windows, and reruns the old
  placement at four times the tokens. Why: App. D, p. 25, predicts that a direction always firing with an existing
  feature is merged into it, which the earlier run's saved decoder showed at 200M tokens.
- **Training length of the diagnostic arm.** The source trains for two epochs of 100M tokens; the diagnostic arm trained
  for 800M. Every state and every other control used the source's 200M.
- **The reading $d$ is ours.** The source reports counts and lists of chat-specific latents, not a share of the change.
- **Scale and data.** A 67M model with a dictionary of 24,576 latents at expansion 32, at block 2 of four, instead of
  Gemma-2-2B with 73,728 latents at layer 13 of 26. The training text is the Pile plus the task's own text instead of
  FineWeb and chat data. The fine-tune-specific candidates are the top 1,667 latents by decoder-norm ratio, the
  toolkit's 5,000 scaled to the dictionary; coefficients are fitted on the toolkit's full 5,000.

**Departure from the filed plan.** Two early checks never ran on the 52 sides: the fraction of variance explained and
the dead share at a side's first snapshot. They ran on the controls. `experiments/crosscoder_fig4/run.py:237`
calls `run_side` with snapshots off, and `src/commute/crosscoder_sides.py:290` then sets no snapshot marks, so the hook
that runs those checks never fires. Only the early stop was lost: the end checks at `run.py:242-243` are stricter, and
every side passed them. No rerun is needed.

## Draft caption for the crosscoder panel of Fig 6

Crosscoder of Minder et al. 2025, their code and recipe, at block 2: $d$, the shared minus the fine-tune-specific part
of the change, one point per fine-tune seed. On every state but one the fine-tune-specific part is near 0 at 200M tokens, the length at which a
plant on an existing feature is still merged, so $d$ is the shared part. Grey: three random updates, 0.532 to 0.611, above most states; star: a second crosscoder seed, 0.038
from the first. The three fine-tune seeds of one task spread by up to 0.41; the two built sides overlap. The planted
control passes with the negative-coefficient filter floored at noise, not under the published rule, where it misses;
placed on an existing feature, it misses at the 200M tokens each state's crosscoder had and passes under both filters
from 400M. The two filters give identical $d$ on every state.

## Not run

- Crosscoders on the real states at four times the tokens, 800M: not run: outside this cell; the user decides (card
  `reports/decisions/2026-11-25_decision_e311-4x-tokens.md`).
- More than one crosscoder seed per state; a second seed on any state other than Runner-up guess.
- TF32, which would have halved training time but departs from the authors' fp32.
- A sweep of dictionary size or sparsity.
- The size-matched sides of the earlier run, which Fig 4 does not have.
- Any test of why a random update reads as more shared than a trained one.

## Provenance

Self control `experiments/crosscoder_fig4/results/stage0a.json`; planted controls `stage0b_random_s0.json`,
`stage0b_random_s1.json`, `stage0b_answer_half_s0.json` and `stage0b_answer_s0_long.json` in the same folder, the last
with its snapshots under `.planted.snapshots`; sides `results/sides/*.json`, one file per side; shard checks
`results/stage2_*_26.json`. Earlier readings `experiments/crosscoder/results/negbeta_rescore.json` and
`experiments/crosscoder/results/run2/stage0b.json`. Every side ran on an H100 in fp32, taking 47.4 to 49.2 minutes;
run 2 used 26 pods. Plotted values `experiments/crosscoder_fig4/figures/crosscoder_fig4.csv`. Ledger beside this file.
