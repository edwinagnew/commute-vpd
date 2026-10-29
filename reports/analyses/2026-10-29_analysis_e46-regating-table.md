# Does the gate gain tell a learned update from one that learned nothing? E46 stage 0, one table in nats

*E46 stage 0, `experiments/regating_table/`, written 2026-10-29. Plan: `reports/plans/2026-10-28_plan_regating-line-reset.md`,
section 2 with Amendment 1 as adopted, the stage 0a stop and its amendment. Red-team review:
`reports/reviews/2026-10-28_review_regating-line-red-team.md`. Filed predictions: `results/predictions.json`. Results:
`results/stage0a.json` from Slurm 42604 and `results/stage0b.json` from Slurm 42675. Notebook: `notebooks/E46_regating_table.ipynb`.
Both jobs ran on the MATS cluster's debug QoS, at no cost.*

## In plain words

On every saved toy fine-tune and on four kinds of fake update, we measured how far the importance function's new values alone raise the stand-in model's prediction of the answer. Real fine-tunes raised it a lot and fake updates did not; a hand-made write that only nudges the model toward the answer landed well below the familiar-behaviour fine-tunes but close to the new-behaviour ones. The test meant to retire this line did not fire, but the random-noise check was too weak to decide, so the next step goes to the user rather than to the larger model.

## What was asked

The user asked on 2026-10-27 whether the change in Γ's importance values after a fine-tune "is genuinely describing how the
model has changed, or just some strange after-effect of it trying to predict a model it isnt trained on". The red team
then showed that the old headline share reads about the same for an update that learned nothing. The reset plan asked
first whether any number computed from Γ's new values survives its own nulls, read in nats on one table. This report is
that table.

One definition is used throughout. Γ's values are computed twice on the same inputs, once from the original model's
activations and once from the fine-tuned model's. The **gate gain** is $g = L(C_0, G_1) - L(C_0, G_0)$. Here $L$ is
the CI-masked model's mean log-odds of the target at the scored position, $C_0$ are the original subcomponents and
$G_0$, $G_1$ are Γ's values from the two models. It is read in the masked stand-in, never in the real fine-tuned model.
The **real gain** is $G = T_1 - T_0$, the full model's change in the same log-odds. A gate gain of 0 means Γ's new
values give the old subcomponents nothing. A value near the real gain means the stand-in reproduces the behaviour
through Γ alone.

## What was tested

**The updates.** Seven toy tasks on the 67M model, each fine-tuned for a few Adam steps, plus one fine-tune on code text:

| task | what the fine-tune teaches | class | seeds |
|---|---|---|---|
| emoticon | after ` :)`, predict `o` | single trigger | 0, 1 |
| which | after ` which`, predict ` was` | single trigger | 0, 1 |
| we | after ` we`, predict ` just` | single trigger | 0, 1 |
| copy table A, B | after a marker and a key word, repeat the key; 32 keys | existing behaviour, by construction | 0, 1 |
| random map table A, B | after a marker and a key word, give a fixed random word; same 32 keys | new behaviour, by construction | 0, 1 |
| code | Pile code text, a fine-tune that learned nothing | non-learning | steps 10, 100 |

The emoticon, which and we fine-tunes are left whole in Δ, the do-nothing state. The copy and random-map fine-tunes are
also read through their least-squares split, so the updated subcomponents have their own readings.

**The nulls, one per question.** Each is a forward pass on a changed copy of the original weights; nothing is trained.

- **A fine-tune to another word.** For the three single-trigger tasks, a fine-tune that teaches ` http` after the same
  trigger. It asks whether any learned update of this size raises the target.
- **The fine-tune's update with its sign flipped.** Same directions, opposite sign. It asks whether Γ follows the
  direction of the change.
- **A random update matched on side effects.** A Gaussian update with the fine-tune's per-matrix norms. It is scaled so
  that the full model's KL to the original, on the non-scored positions, equals the fine-tune's. The scale was capped at
  ten times the fine-tune's norm. It asks whether Γ reacts to a disturbance that points nowhere in particular.
- **A per-key write.** One change to block 1's MLP output matrix, and nothing else. It is the least-norm matrix that maps
  each key's mean input there onto the answer's unembedding row. It is scaled until its real gain matches the
  fine-tune's. It uses no existing machinery by construction and sits in Δ. It asks whether Γ hands the old
  subcomponents a gain whenever the activations start to predict the target. Only the random-map writes are scored. On
  the copy tasks the answer is the key, and input and output embeddings are tied, so the write adds the key's own embedding.

**What each number should read.** At the original weights every gain is 0 by definition. For a learning fine-tune the
real gain is 10 to 21 nats here. The other-target, flipped and random nulls should read a gate gain at or below 0. The
write should read well below the copy fine-tune if Γ is reading existing machinery rather than "the activations now point
at the answer".

**The floor.** For each state, $F = \max(0.1, 2\,\mathrm{SE})$ in nats, with SE the paired standard error over held-out windows
of the per-window gate gain. It is 0.1 for every random row and up to 0.89 for the fine-tunes and writes.

## What we expected, and why

- **The copy and random-map fine-tunes replicate E12.** Take what the updated subcomponents do unmasked. Under Γ's new
  values the copy tasks should keep at least 0.8 of it, and the random maps at most 0.5. E12 read about 0.9 for copy
  and 0.4 for the random maps. This checks that the rebuild matches E12; it is the old share under another name, not
  new evidence.
- **The random update reads within the floor.** E6 found a size-matched random update inert on the emoticon task. Scaling
  it up to the fine-tune's side effects was meant to give the null a fair chance to move Γ. The filed rule passes when
  every task reads within the floor. It fails when any task reads a quarter of the fine-tune's gate gain. Anything in
  between is inconclusive.
- **Anything that lowers the target does not raise the gate gain.** A state whose real gain is not positive should read a
  gate gain no higher than the floor plus 0.1.
- **The write reads well below the copy fine-tune.** Leaning medium. Suppose Γ simply turns on the target's machinery
  whenever the activations predict the target. Then the write would read like the copy fine-tune, at 0.8 or more of its
  gate gain, and the line would be retired.

## Stage 0a: the control

The first run, Slurm 41161, stopped at the ` http` null after 42 seconds. That fine-tune lowered the emoticon target by
9.7 nats, and the filed rule asked for its real gain to stay within a tenth of the fine-tune's on both sides. The cause
was the rule. A fine-tune that learns ` http` after ` :)` must take probability from `o`, so its real gain is negative by
construction. The user chose option A on 2026-10-29: the null passes when it does not *raise* the target. The rerun,
Slurm 42604, then passed every check in 112 seconds.

| check | read | bar | verdict |
|---|---|---|---|
| original-model cells, single-trigger tasks | identical to E18's filed values | within 0.02 nats | pass |
| original-model cells, copy and random-map tasks | largest gap 0.011 nats | within 0.02 nats | pass |
| switching share on E18's exact emoticon state | 0.5972, equal to E18 | 0.597 ± 0.01 | pass |
| cached single-trigger states, full-model log-odds | largest gap 0.036 nats, on which seed 0 | 2% of the filed gain, 0.21 nats there | pass |
| rebuilt copy and random-map states, full-model log-odds | largest gap 0.028 nats | 2% of the filed gain | pass |
| rebuilt copy and random-map states, switching share | largest gap 0.0013 | 0.03 | pass |
| ` http` null on emoticon, real gain over the fine-tune's | −0.72 | at most 0.1, one-sided | pass |
| code fine-tune, row selection and gains | selection identical; gains −0.052, −0.064 against filed −0.052, −0.065 | 0.005 nats | pass |

The other two ` http` nulls read −0.78 and −0.51 of their fine-tunes' gains, reported and not scored.

## Stage 0b: the table

### Does Γ's new gating help the old subcomponents for the fine-tunes but not for the nulls?

![gate gain of each fine-tune against each null on the same task](../../experiments/regating_table/figures/gate_gain.png)

**Reading it.** Yes, with one weak spot. Every fine-tune reads a gate gain of 3.3 to 11.6 nats, the blue circles at the top.
Every update that lowers the target reads a negative gate gain, down to −4.8 nats. Those are the ` http` fine-tunes in
purple and the flipped updates in orange. The random updates, the grey squares, read within 0.16 of zero on all seven tasks. Five of
them sit inside the floor band. Two sit just outside it: emoticon at −0.155 and random map A at +0.118. Both are far below their
fail bars, the red dashes, at 2.0 and 0.94 nats. The green triangles are the per-key writes, which read between the nulls and the
fine-tunes. The caveat is that the random updates are weaker than they look here, which the next figure shows.

Gate gain $g$, real gain $G$, floor $F$ and the full model's off-target KL, in nats; $n$ held-out windows per state.

| state | G | g | F | n | off-target KL |
|---|---|---|---|---|---|
| emoticon seed 0 | 13.43 | 8.02 | 0.31 | 73 | 0.055 |
| emoticon seed 1 | 13.69 | 8.35 | 0.32 | 73 | 0.064 |
| emoticon ` http` | −9.73 | −3.46 | 0.29 | 73 | 0.059 |
| emoticon flipped | −22.19 | −2.76 | 0.28 | 73 | 0.093 |
| emoticon random | −0.18 | −0.155 | 0.10 | 73 | 0.044 |
| which seed 0 | 10.29 | 7.69 | 0.59 | 100 | 0.122 |
| which seed 1 | 10.87 | 8.05 | 0.57 | 100 | 0.106 |
| which ` http` | −8.06 | −3.03 | 0.50 | 100 | 0.152 |
| which flipped | −22.39 | −4.82 | 0.42 | 100 | 0.384 |
| which random | 0.05 | −0.083 | 0.10 | 100 | 0.055 |
| we seed 0 | 12.53 | 3.56 | 0.23 | 106 | 0.090 |
| we seed 1 | 12.19 | 3.30 | 0.19 | 106 | 0.085 |
| we ` http` | −6.37 | −0.86 | 0.21 | 106 | 0.164 |
| we flipped | −14.41 | −1.32 | 0.20 | 106 | 0.121 |
| we random | −0.01 | −0.080 | 0.10 | 106 | 0.046 |
| copy A seed 0 | 13.99 | 11.23 | 0.46 | 97 | 0.189 |
| copy A seed 1 | 14.07 | 10.71 | 0.43 | 97 | 0.181 |
| copy A flipped | −9.98 | −1.49 | 0.27 | 97 | 0.265 |
| copy A random | 0.17 | 0.077 | 0.10 | 97 | 0.070 |
| copy A write | 13.37 | 5.17 | 0.89 | 97 | 0.460 |
| copy B seed 0 | 14.60 | 11.57 | 0.38 | 97 | 0.205 |
| copy B seed 1 | 14.26 | 11.46 | 0.43 | 97 | 0.196 |
| copy B flipped | −8.62 | −1.68 | 0.31 | 97 | 0.256 |
| copy B random | 0.03 | 0.044 | 0.10 | 97 | 0.072 |
| copy B write | 15.27 | 6.71 | 0.66 | 97 | 0.648 |
| random map A seed 0 | 13.73 | 3.75 | 0.51 | 97 | 0.696 |
| random map A seed 1 | 13.16 | 3.50 | 0.49 | 97 | 0.653 |
| random map A flipped | −8.40 | −1.41 | 0.23 | 97 | 0.410 |
| random map A random | 0.15 | 0.118 | 0.10 | 97 | 0.085 |
| random map A write | 13.10 | 1.80 | 0.38 | 97 | 0.286 |
| random map B seed 0 | 20.74 | 6.01 | 0.59 | 97 | 0.097 |
| random map B seed 1 | 20.64 | 5.77 | 0.54 | 97 | 0.104 |
| random map B flipped | −8.12 | −0.86 | 0.24 | 97 | 0.595 |
| random map B random | 0.06 | 0.046 | 0.10 | 97 | 0.107 |
| random map B write | 20.90 | 3.92 | 0.58 | 97 | 1.302 |
| code step 10 | −0.052 | −0.042 | 0.10 | 64 rows | 0.060 |
| code step 100 | −0.064 | −0.050 | 0.10 | 64 rows | 0.079 |

The code fine-tune's off-target KL is read over every position of its 64 held-out rows, not the 40 non-scored positions.

The two random rows outside the floor, in standard errors of the window mean:

| row | g | SE | g over SE | fail bar, 0.25 × seed-0 fine-tune g |
|---|---|---|---|---|
| emoticon random | −0.155 | 0.034 | −4.6 | 2.01 |
| random map A random | +0.118 | 0.047 | 2.5 | 0.94 |

The SE is over held-out windows for one fixed random direction. It says nothing about how a second random direction
would read.

The flipped updates read a gate gain of the same sign as their real gain on all seven tasks. Its size is 0.09 to 0.22 of
the real gain. The ` http` fine-tunes read 0.13 to 0.38 of theirs, also negative.

### Did each random update reach the fine-tune's side effects?

![KL each random update reached against the fine-tune's](../../experiments/regating_table/figures/random_kl.png)

**Reading it.** Only on one task of seven. The filled point, random map B, matched at 6.5 times the fine-tune's norm.
The other six hit the cap at ten times and stopped short. Emoticon reached 0.80 of its target, the closest. Random map A
reached 0.12, the furthest. So on six tasks the random null was a smaller disturbance than the fine-tune, measured by the
full model's KL on positions the task does not score. A null that passes while weaker than its target passes less than
the rule intended.

| task | fine-tune's KL, the target | random update's KL | reached over target | scale | how it ended |
|---|---|---|---|---|---|
| emoticon | 0.055 | 0.044 | 0.80 | 10 | cap |
| which | 0.122 | 0.055 | 0.46 | 10 | cap |
| we | 0.090 | 0.046 | 0.51 | 10 | cap |
| copy A | 0.189 | 0.070 | 0.37 | 10 | cap |
| copy B | 0.205 | 0.072 | 0.35 | 10 | cap |
| random map A | 0.696 | 0.085 | 0.12 | 10 | cap |
| random map B | 0.097 | 0.107 | 1.10 | 6.50 | matched |

Over the last two grid points, scales 5.3 and 10, the KL grew about as the square of the scale on every task. If that
held, matching would have needed scales of roughly 11 to 29 times the fine-tune's norm. That is an extrapolation, not a
measurement. At the original weights the same reading is about 0.0004 nats, the bf16 noise of the KL itself.

The random updates barely touch the target: their real gains are −0.18 to +0.17 nats. That is by design; they were
matched on side effects, not on the gain.

### Does Γ treat a write that only points the activations at the answer like the copy fine-tune?

![per-key write's gate gain over the copy fine-tune's, against the pass and retire lines](../../experiments/regating_table/figures/write_rho.png)

**Reading it.** No, but Γ does react to it. On random map A the write's gate gain was 0.16 of the copy fine-tune's,
and on random map B it was 0.34. Both sit below the pass line at 0.5, so the retire test did not fire. The blue squares
put this in proportion: the real random-map fine-tunes read 0.31 to 0.52 of the copy fine-tune. So the write reads below
the new-behaviour fine-tune, not far below it. In nats the write gave the old subcomponents 1.8 and 3.9 nats. That is half and two-thirds of
what the random-map fine-tunes gave them. A write that uses no existing machinery still earns a gate gain well above the floor.

| table | write's G, against its target | write's g | write over copy fine-tune | random-map fine-tune over copy fine-tune, seeds 0 and 1 | write's off-target KL against the random-map fine-tune's |
|---|---|---|---|---|---|
| A | 13.10 against 13.73 | 1.80 | 0.160 | 0.33, 0.31 | 0.29 against 0.70 |
| B | 20.90 against 20.74 | 3.92 | 0.338 | 0.52, 0.50 | 1.30 against 0.10 |

The copy writes, reported and not scored, read 0.46 on table A and 0.58 on table B, the second inside the inconclusive
band. Each adds the key's own embedding at block 1, so a high reading there does not separate "existing machinery" from
"the activations point at the answer".

The write is well fitted on single held-out windows, not only on the 32 key means it was solved on. Over 97 held-out
windows the cosine between the write's output and the answer's unembedding row averaged 0.99 on every task, with a worst
case of 0.94. The mean relative error was 0.16 to 0.17, and no write was flagged as ill-posed. The scale needed was
13 to 26.

### Do the copy and random-map fine-tunes replicate E12?

Yes, on every state. There is no figure, since this checks the rebuild and carries no new claim. Take what the updated
subcomponents do unmasked: under Γ's new values the copy states keep 0.87 to 0.97 of it. The random maps keep 0.36 to 0.41.

| state | piece gain U₁ − U₀ | gated piece gain L(C₁,G₁) − L(C₀,G₀) | ratio | bar | E12's filed ratio |
|---|---|---|---|---|---|
| copy A seed 0 | 12.20 | 11.40 | 0.934 | ≥ 0.8 | 0.933 |
| copy A seed 1 | 12.45 | 10.85 | 0.872 | ≥ 0.8 | 0.872 |
| copy B seed 0 | 12.59 | 11.72 | 0.931 | ≥ 0.8 | 0.929 |
| copy B seed 1 | 12.00 | 11.61 | 0.968 | ≥ 0.8 | 0.969 |
| random map A seed 0 | 11.19 | 4.07 | 0.364 | ≤ 0.5 | 0.364 |
| random map A seed 1 | 10.76 | 3.83 | 0.356 | ≤ 0.5 | 0.355 |
| random map B seed 0 | 15.75 | 6.52 | 0.414 | ≤ 0.5 | 0.414 |
| random map B seed 1 | 15.92 | 6.26 | 0.394 | ≤ 0.5 | 0.393 |

The gate gain alone separates the classes as well: 10.7 to 11.6 nats for copy, 3.5 to 6.0 for the random maps. The
updated subcomponents add 0.14 to 0.51 nats on top under Γ's new values, in both classes.

## What each number is blind to, and its positive control

| number | blind to | positive control | noise floor |
|---|---|---|---|
| gate gain g | anything the real fine-tuned model does that the masked stand-in does not; whether Γ's reading is a leak of the answer or a description of a mechanism | the fine-tunes, 3.3 to 11.6 nats; identity check on E18's exact state | max(0.1, 2 SE) per state, 0.10 to 0.89 |
| gated piece gain over piece gain | a behaviour Γ can already switch on but the subcomponents cannot express; it is E12's share under another name | E12's filed ratios, reproduced within 0.002 | E12's two seeds differ by up to 0.06 |
| random update, KL-matched | a structured direction of the same size; one random seed per task; six of seven rows were weaker than the fine-tune | none run: no update known to move Γ only by its size | 0.1 nats |
| per-key write | writes at other layers or of other forms; one write per task | the copy write, which uses existing machinery through the tied embedding | the write's own floor, 0.38 to 0.89 |
| flipped update | nothing it is meant to test beyond smoothness; unscored | — | — |

## Amendments in force

1. **Amendment 1, adopted by the user on 2026-10-28.** Everything in nats. The old switching share was dropped as a
   prediction because every null's real gain is near zero. The Frobenius-matched random update became a KL-matched one.
   The flipped update became an unscored sign check. The per-key write was added as the test that can retire the line.
   Stage 0a's tolerances were set from E38's measured drift.
2. **Before any run, after the code review of 2026-10-28.** Each write reports how it fits on held-out windows. The scale
   search reads a 12-point log grid before bisecting. Stage 0b refuses to run on code changed since stage 0a.
3. **After the stage 0a stop, the user's choice A on both questions, 2026-10-29.** The ` http` null is one-sided: it
   passes when it does not raise the target. The write test is scored on the random-map writes, against the copy
   fine-tune of the same table size; the copy writes are reported only. Fable's proposed check of the cosine between the
   write's and the fine-tune's change to the residual stream was not added.

## What stage 0 closes from the red team, and what it does not

**Closed or narrowed.**

- *The share reads the same for a non-learning or other-target update.* In nats that is no longer so. The non-learning
  code fine-tune reads a gate gain of −0.04 to −0.05, the size of its real gain. The other-target fine-tunes read
  negative gate gains. The share's problem was its near-zero denominator, not the gate gain itself.
- *Any diffuse update reads as Γ switching subcomponents on.* A random update of up to ten times the fine-tune's size
  does not move the gate gain beyond 0.16 nats. This is weaker than it sounds, because six of seven random updates
  had smaller side effects than the fine-tune.
- *Γ turns on a token's machinery whenever the activations predict the token.* Partly. A write that makes the
  activations predict the answer gets a third or less of the copy fine-tune's gate gain. But it still gets half to two-thirds of
  what the new-behaviour fine-tune gets, so the gap between "new behaviour" and "a write that uses no machinery" is
  small.

**Not closed.**

- *Γ is not on the fine-tuned model's causal path.* Every gate gain here is read in the masked stand-in. Stage 0 has
  no causal reading in the real model.
- *Leak or mechanism.* None of the red team's decisive tests ran. They are: Γ's shifted values producing the answer out of
  context, a Γ from another decomposition seed, a Γ refitted on the fine-tuned model.
- *Toy single-position tasks on one model.* Stage 0 is still the 67M model, with one scored position per task. Stage 1,
  on the Qwen line's real updates, is the plan's answer to this.
- *Replication.* Fine-tunes have two seeds. Every null has one: one random direction, one write, one flipped update per
  task.
- *E12's class separation, two tasks per class.* Stage 0 rebuilt and reproduced it; it added no third task.

## What we learned

Read in nats, the gate gain rises only for updates that raise the target and stays at or below zero for every update
that does not. That answers the red team's sharpest complaint about the old share, which came from dividing by a near-zero
real gain. The gate gain is still not a clean reading of "existing machinery": a write that uses none earns 1.8 to 3.9
nats, not far below a fine-tune that teaches a new mapping. For `GOALS.md`'s fork between strengthening existing
behaviour and teaching new, the copy fine-tunes still stand apart from both, but the new-behaviour fine-tunes and the
no-machinery write sit close together. Whether the random null is strong enough to clear the size question stays open.

## How compelling is the evidence

- The control passed on every task, and E12's ratios were reproduced within 0.002.
- The random null is the weak link. It reached the fine-tune's side effects on one task of seven, and two rows sat
  outside the floor by 2.5 and 4.6 standard errors of the window mean. In size they are 0.08 and 0.13 of their fail bars.
- Every null has one seed. The floor is window-sampling noise for one fixed update, not the spread across random draws.
- All tasks are single-position toys on the 67M model: three single-trigger tasks, two copy and two random-map tables,
  and one non-learning code fine-tune.

## Surprises

- **The per-key write earns a real gate gain.** At 1.8 and 3.9 nats it is far above the floor. That is about half to two-thirds of the
  random-map fine-tunes'. No prediction was filed on its size against the random-map fine-tune; the filed test compared it to
  the copy fine-tune only.
- **The random update hit its cap on six tasks of seven.** The plan filed the cap at ten
  times as a guard and gave no expected count of capped rows.
- The emoticon random row reads a small *negative* gate gain, −0.155. It lowers the target, which the non-positive rule
  allows; it is outside the floor in size only.

## Advice and decision

*Placeholder for the lead.*

## Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| 0a | apparatus reproduces E18, E12 and E36 | cells within 0.02; identity 0.597 ± 0.01; T₁ within 2% of gain; E12 r within 0.03; ` http` null ≤ 0.1 of gain, one-sided | all within; null −0.72 | pass |
| P0.2 | copy and random-map fine-tunes replicate E12 | gated piece gain over piece gain ≥ 0.8 copy, ≤ 0.5 random, both seeds | 0.87–0.97 copy; 0.36–0.41 random | pass |
| P0.3 | KL-matched random update moves no gate gain | pass all \|g\| ≤ F; fail any \|g\| > F and ≥ 0.25 × \|g of fine-tune\|; else inconclusive | 5 of 7 within F; emoticon −0.155, random map A +0.118 over F = 0.1, against bars 2.0, 0.94; cap bound on 6 of 7 | inconclusive |
| non-positive rule | updates that do not raise the target do not raise g | g ≤ 0.1 + F whenever G ≤ 0 | every such row negative or within | pass |
| P0.4 | a write with no machinery does not read like the copy fine-tune | random-map writes over copy fine-tune: both ≤ 0.5 pass, either ≥ 0.8 retire | 0.160, 0.338 | pass |
| sign row | flipped update's g has G's sign | unscored | same sign on 7 of 7 | — |
| tree | | P0.2 pass; P0.3 inconclusive; P0.4 pass → card | | card |

The plan's tree has no branch for "P0.3 inconclusive with P0.4 pass" other than a card with the numbers. Neither stage 1
nor retirement is decided here.

## Not run

- A second random direction per task, and any random update past ten times the fine-tune's norm.
- A structured null of the fine-tune's size that is not the fine-tune: for instance its update rotated within each matrix.
- The cosine between the write's and the fine-tune's change to the residual stream after block 1, which Fable proposed.
- Writes at other layers or of other forms; more than one write per task.
- E36's lowercase fine-tune, at the floor; E42's teacher-training states, which no prediction reads.
- The masked models' KL to the original; only the full model's off-target KL was read.
- The old switching-share prediction, dropped by Amendment 1.
- Any reading in the real fine-tuned model, any Γ from another seed or refitted, and anything on Qwen.

## Provenance

- Stage 0a: Slurm 41161 stopped at the null, 42 s, `cluster/logs/commute-regating-table-41161.out`. Rerun after the
  amendment: Slurm 42604, 112 s, commit `7d5fea6`, `results/stage0a.json`, log `cluster/logs/commute-regating-table-42604.out`.
- Stage 0b: Slurm 42675, 109 s of measurement, commit `910effa`, `results/stage0b.json`, log
  `cluster/logs/commute-regating-table-42675.out`.
- MATS cluster, debug QoS, $0. Large state files: `results/large/`, a link into `~/commute-data/results-large/regating_table`.
- Figures: `experiments/regating_table/plot.py`, run through `fleet/cpu-run`, into `experiments/regating_table/figures/`.
