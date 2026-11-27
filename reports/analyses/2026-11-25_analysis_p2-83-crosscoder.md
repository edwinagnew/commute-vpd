# What does Minder et al.'s crosscoder read on Fig 4's states, once its planted control works?

*2026-11-25, updated 2026-11-27 with run 3, baseline-lead. Cell P2.83, experiments E311 (runs 1 and 2, 200M tokens)
and E313 (run 3, every side retrained at 800M tokens). Descriptive: the states are named as built and no state is
called reused or new. Plans `reports/plans/2026-11-25_plan_p2-83-crosscoder-fix.md` with its appendix and card, and
`reports/plans/2026-11-26_plan_p2-83-run3.md` with its card; filed expectations
`experiments/crosscoder_fig4/results/predictions.json`; code `experiments/crosscoder_fig4/run.py` and
`src/commute/crosscoder_sides.py`; figures `experiments/crosscoder_fig4/figures/crosscoder_fig4.png` (200M) and
`crosscoder_fig4_lengths.png` (both lengths).*

We trained the crosscoder of Minder et al. 2025, with their code and settings, between the original 67M model and each
of the 48 fine-tuned states that Fig 4 plots, on three random weight changes, and on planted changes whose answer we
know, first at the standard 200M training tokens and then again at four times that. The planted change is found at both
lengths, but on the real fine-tunes the fine-tune-specific part stays near zero at four times the training, a random
update still reads as more shared than most of them, and the two built sides still overlap. For `GOALS.md` the
crosscoder offers no reading of reuse against new on this model, and the reason is not too little training, up to 800M tokens.

![Left: the crosscoder's reading on each of Fig 4's states, three fine-tune seeds a task, at 200M tokens (black) beside 800M tokens (orange), with the random updates as their own column on the same axes. Right: the planted control at both lengths, literal filter (filled) beside floored filter (hollow), against the bar of one half; the every-window plant at 200M has the literal reading only.](../../experiments/crosscoder_fig4/figures/crosscoder_fig4_lengths.png)

The 200M-only figure of the earlier run, with the second crosscoder seed, is `experiments/crosscoder_fig4/figures/crosscoder_fig4.png`.

## The answer

The crosscoder can find a planted new direction on this model, under the authors' rule when the plant sits on an
existing feature and with their sign filter floored when it does not. On the real fine-tunes, retraining every
crosscoder for four times as long changed little. Almost none of the change sits on fine-tune-specific latents at
either length, a random update reads more shared than most fine-tunes, and the tasks built to reuse and built new
overlap. Training length does not explain the crosscoder's silence here.

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
train theirs on a 26-layer model. In runs 1 and 2 each crosscoder saw 200M tokens, two epochs of 100M as in the source.
Run 3 retrained every one of them on the same 100M tokens for eight epochs, 800M tokens. One crosscoder was trained for
each side:

- **48 states**: the sixteen tasks of Fig 4, eight built to reuse the original model's machinery and eight built new,
  each fine-tuned from three seeds.
- **Three random updates**: a Gaussian weight change at the per-matrix sizes of each seed of the Runner-up guess task.
  A random update teaches nothing, so it shows what the tool reads when no task was learnt.
- **One second crosscoder seed** on the seed-0 Runner-up guess state, at 200M only, to show how much of the spread
  comes from the crosscoder itself rather than from the fine-tunes.

Two controls ran first, alone, at each length. The **self control** trains a crosscoder of the original model against
itself; it should find every latent shared and none specific. The **planted control** adds one fixed random direction
to the fine-tuned side's activations only, so the right answer is known: one latent should hold that direction, be
called fine-tune-specific, and carry at least half the planted change.

The planted control of an earlier run of this crosscoder missed. That run put the direction at the answer position of
every task window, where existing features of the original model always fire. The saved crosscoder had not given the
plant its own latent; it had written the plant into the fine-tuned decoder of the latent that already fires there, and
by the authors' own test such a latent is shared. Runs 1 and 2 test two explanations. One variant moves the plant to
positions chosen without regard to content, one in every 80, with two crosscoder seeds; another keeps the answer
position but plants only half the windows. A diagnostic arm keeps the earlier placement exactly and trains on 800M
tokens, to see whether the merged latent separates with more training. When it did, run 3 asked the same of the real
states: a redteam of runs 1 and 2 had warned that the near-zero fine-tune-specific part at 200M might be training length.

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
- The 48 states at 200M: seed-0 states within 0.05 of the earlier run's readings, and the two crosscoder seeds about
  0.05 apart. No bar was filed on the states; the reading is descriptive.
- The 48 states at 800M, from run 3's card, which named two rival outcomes and chose neither. If the 200M silence was
  training length, the tasks built new should show a fine-tune-specific part of at least 0.05 on most seeds. If the
  fine-tunes change block 2 inside directions the original latents already use, that part should stay under 0.05.

## Does the planted control pass?

| control | tokens | filter | lead latent: error ratio, reconstruction ratio | lead latent: original-model reconstruction coefficient | part on specific latents | pass |
|---|---|---|---|---|---|---|
| self, original against itself | 200M | both | none specific | none specific | none | yes |
| self, original against itself | 800M | both | none specific | none specific | none | yes |
| content-independent, crosscoder seed 0 | 200M | literal | 0.088, −1.3e-4; dropped | −3.1e-4, under 0 | 0 | no |
| content-independent, crosscoder seed 0 | 200M | floored | 0.088, −1.3e-4; kept | −3.1e-4, above its bar −0.0070 | 1.00 | yes |
| content-independent, crosscoder seed 1 | 200M | literal | 0.023, −1.6e-4; dropped | −2.1e-4, under 0 | 0.298 | no |
| content-independent, crosscoder seed 1 | 200M | floored | 0.023, −1.6e-4; kept | −2.1e-4, above its bar −0.0042 | 0.999 | yes |
| content-independent, crosscoder seed 0 | 800M | literal | 0.040, −1.3e-5; dropped | −2.4e-5, under 0 | 0 | no |
| content-independent, crosscoder seed 0 | 800M | floored | 0.040, −1.3e-5; kept | −2.4e-5, above its bar −0.0055 | 0.997 | yes |
| answer position, half the windows | 200M | literal | 0.112, 0.063 | positive | 0.737 | yes |
| answer position, half the windows | 200M | floored | 0.112, 0.063 | positive | 0.737 | yes |
| answer position, half the windows | 800M | literal | 0.062, 0.071 | positive | 0.933 | yes |
| answer position, half the windows | 800M | floored | 0.062, 0.071 | positive | 0.933 | yes |
| answer position, every window, earlier run | 200M | literal | 0.244, 0.089; error ratio over the cut-off 0.2 | not saved | 0 | no |
| answer position, every window, this run | 800M | literal | 0.120, 0.076 | 0.088, positive | 0.804 | yes |
| answer position, every window, this run | 800M | floored | 0.120, 0.076 | 0.088, positive | 0.804 | yes |

**Reading it.** Under the authors' published rule, the error-ratio cut-off of 0.2 and our bar of one half on the
specific part, the two plants at the answer position pass at 800M. The half-window plant also passes at 200M. The
content-independent plant misses under the literal filter at both lengths and passes under the floored one at both. In
every crosscoder at 200M a single latent took that plant (two latents at 800M under the floored filter), with a decoder-norm ratio of 0.85 and 0.93 toward the fine-tuned
side at 200M. Its error ratio was 0.088 and 0.023 at 200M and 0.040 at 800M, well under the authors' cut-off. The
literal filter drops it only because its original-model reconstruction coefficient came out negative. That
coefficient was −3.1e-4 at crosscoder seed 0, against a floored bar of −0.0070, and −2.1e-4 at seed 1, against a bar of
−0.0042. At 800M it was −2.4e-5, against a bar of −0.0055. All three are noise around zero.

The coin toss shows inside one run. Across the 800M run's snapshots the literal reading of this plant went from 0 to
0.537 and back to 0, while the floored reading stayed at 0.823 or above from 50M tokens on. So the cell's planted
control passes under the published rule wherever the lead latent's coefficients came out positive, and the one miss
under that rule is shown to come from its sign filter acting on noise, not from a plant the crosscoder cannot find. The
self control passed every check at both lengths: no specific latent, every ratio exactly one half, and a fraction of
variance explained of 0.932 at 200M and 0.937 at 800M.

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
decide whether the plant is found. This is why run 3 retrained the real states at 800M.

## Does more training change the fine-tune-specific part on the real states?

Hardly. At both lengths the fine-tune-specific part is near zero on the real states, on the tasks built new as much as
on those built to reuse, though it is not exactly zero everywhere. At 200M its largest value is 0.101, on the seed-0
state of Copy key (96 keys), and on every other state it is at most 0.005. At 800M that state's part fell to 3.2e-4,
and its $d$ rose from 0.078 to 0.164. The nonzero values at 800M, all others being exactly 0:

| task | built | seed | part on fine-tune-specific latents at 800M |
|---|---|---|---|
| Copy key | reuse | 1 | 0.019 |
| Copy key | reuse | 2 | 0.0011 |
| Copy key (96 keys) | reuse | 0 | 3.2e-4 |
| Cipher (96 keys) | new | 0 | 0.018 |
| Cipher (96 keys) | new | 1 | 0.0069 |
| Cipher (96 keys) | new | 2 | 0.020 |
| Cipher (late step) | new | 1 | 8.0e-5 |
| Cipher (late step) | new | 2 | 2.6e-4 |
| Prior-matched cipher | new | 0 | 0.020 |
| Prior-matched cipher | new | 1 | 9.2e-4 |

**Reading it.** The redteam's caveat is tested and does not hold. Four times the training leaves the largest
fine-tune-specific part at 0.020, under the 0.05 that the training-length outcome expected on most seeds of the tasks
built new. The values sit on some of the cipher tasks built new and on Copy key, which is built to reuse; the other
tasks built new, Ciphered text, Lookup, Random map, Cipher (24 keys) and Cipher (early step), read exactly 0 at 800M.
With so little on fine-tune-specific latents, $d$ is in effect the shared part alone at either length.

## Does the choice of filter change any reading on the real states?

No. At 200M, on all 52 sides, the 48 states, the three random updates and the second crosscoder seed, the two filters
give the same $d$ to every digit stored. At 800M they agree to every stored digit on every side except seed 2 of Cipher
(96 keys), where $d$ reads 0.3292 under the literal filter and 0.3288 under the floored one: the floored filter keeps
one more latent, raising the fine-tune-specific part from 0.0196 to 0.0200. With almost no specific latents to drop,
the filter has nothing to act on.

The floored filter has its own blind spot. Its bar needs a standard error, and a latent that fires on fewer than ten
tokens in some fold gets none; for such a latent the floored filter falls back to the literal one. On the seed-0 state
of Cipher (24 keys) at 200M that happened for 1,893 of the 5,000 latents given coefficients, and in the 800M planted run
for 3,529 of 5,000. It changes no reading here: the filters differ on one side by 0.0004, and every planted control's
lead latent had a defined bar. The 5,000 are the wider candidate set. Coefficients are fitted for the top 5,000 latents
by decoder-norm ratio, the diffing toolkit's number. The fine-tune-specific group in $d$ is taken from the top 1,667,
that number scaled to our dictionary, and the reading on the wider set is stored beside it.

## What does the crosscoder read on the 48 states?

The random updates read 0.532, 0.550 and 0.611 at 200M, and 0.540, 0.575 and 0.665 at 800M. Values are $d$ under the
literal filter, one per fine-tune seed; task names are Fig 4's.

| task | built | 200M seed 0 | 200M seed 1 | 200M seed 2 | 800M seed 0 | 800M seed 1 | 800M seed 2 |
|---|---|---|---|---|---|---|---|
| Copy at a new marker | reuse | 0.567 | 0.531 | 0.613 | 0.573 | 0.517 | 0.622 |
| Copy key | reuse | 0.373 | 0.287 | 0.378 | 0.343 | 0.245 | 0.369 |
| Copy key (24 keys) | reuse | 0.379 | 0.405 | 0.470 | 0.487 | 0.534 | 0.406 |
| Copy key (96 keys) | reuse | 0.078 | 0.422 | 0.489 | 0.164 | 0.360 | 0.464 |
| Lowercase text | reuse | 0.389 | 0.447 | 0.513 | 0.434 | 0.513 | 0.486 |
| Runner-up guess | reuse | 0.398 | 0.436 | 0.240 | 0.416 | 0.452 | 0.084 |
| Top guess | reuse | 0.360 | 0.404 | 0.451 | 0.305 | 0.381 | 0.456 |
| Top guess (96 keys) | reuse | 0.506 | 0.553 | 0.157 | 0.504 | 0.535 | 0.125 |
| Cipher (24 keys) | new | 0.298 | 0.375 | 0.402 | 0.276 | 0.348 | 0.369 |
| Cipher (96 keys) | new | 0.478 | 0.401 | 0.372 | 0.419 | 0.304 | 0.329 |
| Cipher (early step) | new | 0.388 | 0.249 | 0.204 | 0.298 | 0.217 | 0.263 |
| Cipher (late step) | new | 0.597 | 0.480 | 0.520 | 0.589 | 0.410 | 0.503 |
| Ciphered text | new | 0.519 | 0.602 | 0.547 | 0.541 | 0.548 | 0.545 |
| Lookup | new | 0.485 | 0.632 | 0.606 | 0.454 | 0.621 | 0.462 |
| Prior-matched cipher | new | 0.391 | 0.473 | 0.506 | 0.333 | 0.449 | 0.497 |
| Random map | new | 0.407 | 0.288 | 0.381 | 0.362 | 0.140 | 0.330 |

**Reading it.** At both lengths most states read below the random updates, not above them. The 48 states average
0.426 at 200M and 0.404 at 800M, while the random updates run from 0.532 to 0.611 and from 0.540 to 0.665. At 800M the
states that reach the random updates' range are every seed of Ciphered text, seeds 0 and 2 of Copy at a new marker,
seed 1 of Lookup and seed 0 of Cipher (late step); none reads above it. A random update that taught nothing thus reads more
shared than most real fine-tunes. A high $d$ is therefore no evidence of reuse, and the random updates are the tool's
value for "no task learnt", not a floor and not a reference for new. Why a random change reads this way was not tested.

The crosscoder cannot order the two built sides at either length. The tasks built to reuse average 0.410 at 200M and
0.407 at 800M; the tasks built new average 0.442 and 0.400. Their values overlap across the whole range. At 200M the
high end on both sides was the two halves of one built pair, Copy at a new marker at 0.570 and Lookup at 0.574. At
800M Copy at a new marker still averages 0.570, then come Ciphered text at 0.545 and Lookup at 0.512.

Tasks can still differ by more than their seeds spread. Copy key reads 0.245 to 0.369 at 800M and Copy at a new marker
0.517 to 0.622, so those two do not overlap at either length. But one task's seeds can spread as widely as tasks
differ: at 800M Top guess (96 keys) runs from 0.125 to 0.535, a spread of 0.410, and Runner-up guess from 0.084 to
0.452. At 200M Copy key (96 keys) ran from 0.078 to 0.489, a spread of 0.411.

Single states moved between the lengths by more than the crosscoder's own noise. Seed 2 of Runner-up guess fell from
0.240 to 0.084, now the lowest of all sides, and seed 1 of Copy key (24 keys) rose from 0.405 to 0.534. The crosscoder's
own noise comes from reseeding it. At 200M a second crosscoder seed on the seed-0 Runner-up guess state read 0.437
against the first seed's 0.398, a gap of 0.038. The earlier run's two crosscoder seeds on that state, 0.416 and 0.463,
have a standard deviation of 0.033. There is no second crosscoder seed at 800M. The moves did not change the picture:
below the random updates, sides overlapping, seeds spreading as widely as tasks.

## Against the earlier run of the same recipe

Most seed-0 states on the key-value tasks read the same at 200M as the earlier run to three decimals. Two moved, both
by less than 0.05. Cipher (24 keys) read 0.298 against 0.266 then. Runner-up guess read 0.398 against 0.416. The
earlier run's second crosscoder seed had read 0.463, a gap of 0.047 against this run's 0.038. The text-task states
moved more, as expected: their crosscoders now train on the full token budget, where the earlier ones ran out at about
half. The seed-0 Lookup state read 0.485 against 0.549 then. The seed-0 Ciphered text state read 0.519 against 0.563.
The seed-0 random update read 0.550 against 0.547.

## What we learned

The crosscoder can isolate a fine-tune-only direction at block 2 of the 67M. The earlier misses came from too little
training for a plant that rides on an existing feature, and from the sign filter, not from a method that cannot find
the plant. On real fine-tunes, though, it finds almost no fine-tune-specific part at 200M or at 800M tokens. At 200M
that part is 0.101 on the seed-0 state of Copy key (96 keys) and at most 0.005 elsewhere; at 800M it is at most 0.020,
on some cipher tasks built new and on Copy key, and exactly 0 on the rest. So the reading reduces to the shared part of
the change, and four times the training leaves it where it was: higher for a random update than for most trained ones,
with one task's seeds spreading as widely as tasks differ and the two built sides overlapping. For `GOALS.md` the
crosscoder joins the other model-diffing tools as a baseline whose values on Fig 4's states overlap between the two
built sides; it gives no reading of reuse against new on this model, and longer training, up to 800M tokens, does not change that.

## How compelling is the evidence

- The control is strong for what it tests: one fixed direction added only to the fine-tuned side, found at two
  crosscoder seeds, at three placements and at both lengths, with most of its change on one latent. It does not test a
  change spread across many directions, which is what a fine-tune makes.
- The training-length caveat is tested up to 800M, the length at which every planted control passes. A length beyond
  800M was not tried.
- The reading is blind to whatever the crosscoder puts in the bias, in latents that fail latent scaling, and in the
  error; on the states the bias alone averages about a fifth of the change.
- Every state has three fine-tune seeds and one crosscoder per length. Only one state has a second crosscoder seed, at
  200M, so the crosscoder's own noise is one gap of 0.038, with the earlier run's standard deviation of 0.033 beside
  it, against fine-tune seed spreads up to 0.411 within a task.
- All sides passed their end checks at both lengths: the fraction of variance explained was at least 0.927 at 200M
  and 0.932 at 800M, against bars of 0.882 and 0.887, and no side had dead latents against the bar of 0.3.

## Surprises

- The earlier placement passes with four times the training. The plan's diagnosis had put the earlier miss down to the
  placement alone; the long arm shows it was the placement and the training length together.
- The literal filter drops the content-independent plant's latent at both crosscoder seeds and at both lengths, on a
  coefficient within noise of zero. This was the filed lean, so it is not a reversal, but it means the authors' rule
  as published passes this control here only for the placements whose latent's coefficients came out positive.
- The fine-tune-specific part on real states is almost zero, so $d$ is the shared part alone. That was not filed, and
  it holds at 800M, where the controls had shown that more training separates a merged plant.

## What next

- Nothing is proposed. The reading would not change with more crosscoder seeds, since the crosscoder's own gap is a
  tenth of the fine-tune seed spread, nor, on run 3's evidence, with more training.

## Scorecard

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| 0a | self control, 200M and 800M | no specific latent; ratios within 0.001 of one half | none; exact, at both lengths | as expected |
| 0b random s0, s1 | plant at content-independent positions, 200M | lead specific and part at least 0.5; gate on the floored filter | floored 1.00 and 0.999; literal 0 and 0.298 | floored pass; literal miss, as leaned |
| 0b random s0 long | the same, 800M | as above | floored 0.997; literal 0 | floored pass; literal miss |
| 0b answer_half | plant on half the windows, 200M and 800M | as above | 0.737 and 0.933 under both | pass |
| 0b answer long | earlier placement reproduced at 200M; error ratio under 0.2 by 800M | diagnostic | 0.244 at 200M, equal to the earlier run; 0.120 and part 0.804 at 800M | as leaned; pass under both filters |
| 2 states | seed-0 key-value states within 0.05 of the earlier run | absolute gap | largest gap 0.031 on Cipher (24 keys) | as expected |
| 2 reseed | crosscoder seeds about 0.05 apart | absolute gap | 0.038 | as expected |
| 2 filters | floored $d$ at most the literal $d$ | per side | equal at 200M; at 800M equal except Cipher (96 keys) seed 2, 0.3288 against 0.3292 | as expected |
| run 3 | fine-tune-specific part at 800M: at least 0.05 on most seeds built new, or under 0.05 | descriptive, no bar | at most 0.020; 0 on most states | the under-0.05 outcome |

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
- **Training length.** The source trains for two epochs of 100M tokens. Runs 1 and 2 used that length for every state;
  the diagnostic arm and all of run 3 trained for eight epochs, 800M. Why: the diagnostic arm showed that a merged
  plant separates only after 200M.
- **The reading $d$ is ours.** The source reports counts and lists of chat-specific latents, not a share of the change.
- **Scale and data.** A 67M model with a dictionary of 24,576 latents at expansion 32, at block 2 of four, instead of
  Gemma-2-2B with 73,728 latents at layer 13 of 26. The training text is the Pile plus the task's own text instead of
  FineWeb and chat data. The fine-tune-specific candidates are the top 1,667 latents by decoder-norm ratio, the
  toolkit's 5,000 scaled to the dictionary; coefficients are fitted on the toolkit's full 5,000.

**Departure from the filed plan.** Two early checks never ran on the sides: the fraction of variance explained and the
dead share at a side's first snapshot. They ran on the controls. `experiments/crosscoder_fig4/run.py:237`
calls `run_side` with snapshots off, and `src/commute/crosscoder_sides.py:290` then sets no snapshot marks, so the hook
that runs those checks never fires. Only the early stop was lost: the end checks at `run.py:242-243` are stricter, and
every side passed them at both lengths. No rerun is needed.

## Draft caption for the crosscoder panel of Fig 6

Crosscoder of Minder et al. 2025, their code and recipe, at block 2: $d$, the shared minus the fine-tune-specific part
of the change, one point per fine-tune seed, at 200M and at 800M training tokens. The fine-tune-specific part is near 0
at both lengths: at most 0.005 on every state but one at 200M (0.101, Copy key (96 keys) seed 0) and at most 0.020 at 800M, so
$d$ is the shared part. Random updates read 0.532 to 0.611 at 200M and 0.540 to 0.665 at 800M, above most states. The
three fine-tune seeds of one task spread by up to 0.41; the two built sides overlap at both lengths. The planted
control passes under the published rule at the answer position: on half the windows from 200M tokens, on every window
from 400M. Placed where no feature fires, it passes only with the negative-coefficient filter floored at noise. The two filters give the same $d$ on every
state but one, Cipher (96 keys) seed 2 at 800M, 0.3292 literal and 0.3288 floored.

## Not run

- `e54__elicit_s0__seed1`, the second crosscoder seed on the seed-0 Runner-up guess state, at 800M: not run: the
  user's cap of $800 ("Stay under $800"). Run 3's pods cost $748.26 by the plugin's spend record.
- The content-independent plant at crosscoder seed 1 at 800M.
- A training length beyond 800M.
- More than one crosscoder seed per state; a second seed on any state other than Runner-up guess.
- TF32, which would have halved training time but departs from the authors' fp32.
- A sweep of dictionary size or sparsity.
- The size-matched sides of the earlier run, which Fig 4 does not have.
- Any test of why a random update reads as more shared than a trained one.

## Provenance

200M: self control `experiments/crosscoder_fig4/results/stage0a.json`; planted controls `stage0b_random_s0.json`,
`stage0b_random_s1.json`, `stage0b_answer_half_s0.json` and `stage0b_answer_s0_long.json` in the same folder, the last
an 800M run with its snapshots under `.planted.snapshots`; sides `results/sides/*.json`, one file per side; shard checks
`results/stage2_*_26.json`. 800M, run 3: `results/long/stage0a.json`, `results/long/stage0b_random_s0_long.json`,
`results/long/stage0b_answer_half_s0_long.json`, sides `results/long/sides/*.json`, shard checks
`results/long/stage2_*_26.json`, logs `results/long/run_*.log`. Earlier readings
`experiments/crosscoder/results/negbeta_rescore.json` and `experiments/crosscoder/results/run2/stage0b.json`. Every
side ran on an H100 in fp32: 47.4 to 49.2 minutes at 200M on 26 pods, and 189.6 to 259.8 minutes at 800M. Plotted
values `experiments/crosscoder_fig4/figures/crosscoder_fig4.csv`. Ledger beside this file.
