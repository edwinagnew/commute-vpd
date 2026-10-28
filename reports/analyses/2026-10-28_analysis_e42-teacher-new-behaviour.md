# Analysis: can teacher training copy a behaviour the model lacked, faithfully and with Δ fixed? (E42)

*2026-10-28. Plan `reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md`; predictions
`experiments/teacher_training/results/predictions_e42.json`, filed 11:57, unedited; notebook
`notebooks/E42_teacher_new_behaviour.ipynb`; figures `experiments/teacher_training/figures/e42_{copy,trigger,routes,pile}.png`
from `experiments/teacher_training/plot_e42.py`.*

*Independent check: `reports/reviews/2026-10-28_review_e42-check.md`, corrections applied.*

We trained the decomposition's parts to imitate fine-tunes that teach the small model a word cipher it lacked, holding
the importance network and the leftover weights fixed. The copies kept the cipher at its cue and passed the
reconstruction checks on ordinary text, yet the model built only from the parts marked important at the cue lost almost
all of it. The criterion in force counts this as a pass, and the open question for the user is whether this case
should change that criterion.

## The answer

Yes by the criterion in force, with one caveat that matters. We copied fine-tunes that teach the model a word cipher.
At the cipher's trigger the copies kept the new behaviour on all three tasks. On ordinary text they passed the
harness's reconstruction checks, and Δ stayed untouched. But with Γ frozen, the model built only from the subcomponents
Γ marks important at the trigger loses almost all of it on both random ciphers. No map read so far avoids that.
Unexpectedly, a copy trained on plain Pile text alone reached 0.91 on the one-pair task, where we filed below 0.5.

**A filed prediction reversed.** We filed that the copy trained on Pile text alone would stay below 0.5 on every task.
On the one-pair task it reached 0.91 and passed the check, which the filed rule scores "learnt". That is 0.01 over the
line, so a rerun could read "between", which would still contradict the filing. What a copy can learn from text
without the marker is the fine-tune's key-to-value bigram, which also answers the held-out windows. The fine-tune shows
that bigram in plain text: right after the key word it puts 0.31 on the value, where the unchanged model puts 0.0001.
Whether that is what teaches the copy is untested.

## What we were trying to test

Teacher training is component training whose loss is the fine-tuned model's predictions. The alive subcomponents' read
and write vectors start from the unchanged decomposition and are trained to minimise the KL from the fine-tune to the
decomposition's model, with Γ and Δ frozen. In E34 it kept the behaviour of six fine-tunes and passed the adversarial
check. Each of those fine-tunes taught something the model already half did. `GOALS.md` asks what happens when
post-training teaches something new. Its "create" branch expects that with Δ fixed, the new behaviour runs through
subcomponents Γ keeps off.

The new behaviour is E10's key-to-value cipher. Each window holds a marker token that never occurs in the source text,
then a common word, the key. The fine-tuned model must put the key's value next. Each task has 403 training and 97
held-out windows:

- **32 random pairs**: each of 32 keys maps to an arbitrary other word.
- **1 random pair**: one key and its value.
- **Identity control**: each of the 32 keys is followed by itself. The model half does this, and in E10 its gain came
  from Γ switching on subcomponents that already copy. It separates "the format" from "the behaviour is new".

The fine-tune is E10's recipe run to step 1,000: all 24 matrices, Adam at 3e-5, batch 32, seed 0. It is copied twice
from the unchanged decomposition, with Adam at 1e-5 for 1,000 steps. The copy on task text uses the training windows
with E34's trigger weighting. The copy on Pile text uses 8,192 Pile training rows with a plain mean KL. Those rows hold
no marker; batches are 8 rows of 512 tokens.

Readings per state:

- **Held-out P(target)**, the target's probability at the trigger. The unchanged model reads 0.000015 on 32 pairs and
  0.00017 on the other two, from E10.
- **Pile harness**: the 20-step adversary and the stochastic-mask reading on 256 Pile validation rows, from one
  start. Each is a ratio to the unchanged model in the same job. The verdict needs P ≥ 0.9 first. Then each ratio passes
  at or below $1.5/(1+f)$ and fails at or above $1.5(1+f)$. Here $f$ is the unchanged model's spread over three
  adversary seeds in the same job. On this L40, $f = 0.209$, so the adversarial lines are 1.24 and 1.81.
- **The split of the gain at the trigger.** The gain is the rise in the target's mean log-odds at the trigger over the
  unchanged model. It splits exactly into five shares. Direct: the changed subcomponents at the unchanged model's
  causal-importance values. Then Γ's changed values, their interaction, masked-off, and Δ's. The masked-off share is the
  part present with every subcomponent on but lost under Γ's mask. At 0 the CI-masked model keeps the whole gain; at 1
  it keeps none.
- **Teacher gap**: the trigger-weighted KL from the fine-tune to the state on held-out windows. The unchanged model's
  gap is the reference.
- **Pile reading**: the mean probability of the key's value at every position right after a key word. It is read in
  the first 1,024 Pile-text rows, for the fine-tune and the unchanged model, with counts of the marker.

References: **do-nothing** puts the whole fine-tune in Δ. The **least-squares split** is the least-norm change of the
subcomponents, with the remainder in Δ. Its expected fail shows the check has power on the task.

## What we expected, and why

- **The task-text copy keeps the behaviour and passes on every task**, with high confidence on 32 pairs. Label-trained
  component training passed there once settled. Every task-text copy in E34 passed.
- **Its gain is masked off on both random ciphers**, high confidence: label training read 0.75 and 0.86. On the
  identity control, not masked off, medium confidence: label training read 0.21.
- **The Pile-text copy learns nothing of the cipher**, medium confidence. The text holds no marker. The fine-tune's own
  loss fitted its predictions at the windows' other positions, where keys occur unmarked. The plan also expected the
  fine-tune's Pile reading within 10 times the unchanged model's.

## Controls

| Check | Read | Must read | Verdict |
|---|---|---|---|
| C1, unchanged model, three adversary seeds | KL 0.692 / 0.836 / 0.705; floor 0.209; known-value errors 0.25% and 0.05% | pass; floor about 0.21 on an L40 | pass, identical in all three jobs |
| G1, fine-tune at step 10 vs E10 | 0.58470 / 0.95952 / 0.99824 for rand32 / ident32 / rand1 | within 0.02 of 0.5847 / 0.9595 / 0.9982 | pass ×3; on rand1 it could not fail |
| Gate, fine-tune at step 1,000 | 0.9942 / 0.9972 / 1.0000 | ≥ 0.9 | pass ×3 |
| Power, least-squares split fails | 1.977 / 1.855 / 1.826 | ≥ 1.81 | power on all three; thin on ident32 and rand1 |

The fine-tune at step 300 reads 0.99651 / 0.99953 / 0.99999, matching E10's 0.9965 / 0.9995 / 1.0000.

## Does the copy keep the behaviour and pass the check, with Δ untouched?

![copy keeps the behaviour and passes](../../experiments/teacher_training/figures/e42_copy.png)

**Reading it.** Yes, on all three tasks. Every task-text copy keeps the target at 0.96 or more at the trigger. It
passes the four reconstruction checks the harness runs on 256 Pile validation rows: stochastic, adversarial from one
20-step start, CI-masked and unmasked. VPD's sparsity loss was not read. Under the adversary the copies read 0.96 to
0.99. That is below all three of the unchanged model's adversary seeds, which read 1.00, 1.21 and 1.02 on this scale.
The least-squares split fails on every task, so the check can fail here, though only just on two tasks. Do-nothing
reads between the lines on all three. Δ took none of the copy's change.

The copy is the fine-tune's behaviour at the trigger, not the fine-tune. Away from the trigger it does 0.35 to 0.40
of the fine-tune's damage, and its Pile unmasked ratio is 0.81. Part of why it passes on Pile text is that it moved the
model less there than the fine-tune did. The teacher gap's "under 1% of the unchanged model's" is dominated by the
trigger term, so it does not show fidelity away from the trigger.

Step 1,000 under the Pile harness, as ratios to the unchanged model. The stochastic lines are 1.497 and 1.503.

| task | state | P(target) | adversarial | stochastic | verdict | teacher gap, nats (W₀'s) | damage elsewhere | size | Δ's share |
|---|---|---|---|---|---|---|---|---|---|
| rand32 | do-nothing | 0.994 | 1.499 | 0.892 | inconclusive | 0 (13.19) | 0.0607 | 0.0379 | 1 |
| rand32 | least-squares | 0.994 | 1.977 | 0.898 | fail | 0 | 0.0607 | 0.0379 | 0.369 |
| rand32 | task-text copy | 0.964 | 0.960 | 0.936 | **pass** | 0.070 | 0.0243 | 0.0262 | 5e-10 |
| rand1 | do-nothing | 1.000 | 1.557 | 0.899 | inconclusive | 0 (10.49) | 0.0583 | 0.0352 | 1 |
| rand1 | least-squares | 1.000 | 1.826 | 0.905 | fail | 0 | 0.0583 | 0.0352 | 0.369 |
| rand1 | task-text copy | 1.000 | 0.989 | 0.941 | **pass** | 0.024 | 0.0204 | 0.0175 | 1e-9 |
| ident32 | do-nothing | 0.997 | 1.459 | 0.887 | inconclusive | 0 (10.19) | 0.0597 | 0.0355 | 1 |
| ident32 | least-squares | 0.997 | 1.855 | 0.893 | fail | 0 | 0.0597 | 0.0355 | 0.372 |
| ident32 | task-text copy | 0.983 | 0.957 | 0.929 | **pass** | 0.048 | 0.0226 | 0.0210 | 7e-10 |

At step 100, read without the harness, the task-text copies already hold 0.970, 1.000 and 0.999. Label-trained
component training on rand32 read adversarial ratios of 0.975 and 0.944 in E10 Amendment 1. The copy's 0.960 sits
between them.

## Does the CI-masked model carry the new behaviour?

![CI-masked model on Pile text and at the trigger](../../experiments/teacher_training/figures/e42_trigger.png)

**Reading it.** On ordinary text yes; at the trigger no, on both random ciphers. On Pile text the CI-masked KL of
every state sits within 4% of the unchanged model's. At the trigger, the copy on 32 pairs puts 0.96 on the target with
Δ included. With every subcomponent on and Δ out it puts 0.95; under Γ's mask, 0.007. On one pair the masked model keeps
0.001 of 1.00. On the identity control it keeps 0.93 of 0.98. These are means over 97 held-out windows, from one
batch order. In log-odds the masked model still keeps 14% and 5% of the gain on the two random ciphers, so "almost all"
holds in probability, not in log-odds.

With Γ frozen, no map read so far keeps the cipher in the CI-masked model at the trigger. Teacher training and E10's
label-trained component training carry it through subcomponents Γ keeps off there. Do-nothing leaves it in Δ, and the
least-squares split leaves much of it in Δ. So the gap is not specific to teacher training. Whether any frozen-Γ map
can close it is untested: teacher training with VPD's masked losses on the task's windows has not been tried here.
The Pile checks are unlikely to see this. The marker did not occur in the 1,024 Pile training rows scanned, and it was
chosen for zero count in the windows' source rows. The harness's 256 validation rows were not scanned.

Trigger cells come from the run logs' route-split lines, to three decimals; the results files keep only the shares.
"Full" is the state's own model with Δ, "unmasked" has every subcomponent on and Δ out, and "CI-masked" has Γ's mask
and Δ out. Pile ratios are to the unchanged model.

| task | state | full P | unmasked P | CI-masked P | Pile CI-masked ratio | Pile unmasked ratio |
|---|---|---|---|---|---|---|
| rand32 | do-nothing | 0.994 | 0.000 | 0.016 | 0.988 | 8.663 |
| rand32 | least-squares | 0.994 | 0.221 | 0.017 | 0.995 | 1.208 |
| rand32 | task-text copy | 0.964 | 0.952 | **0.007** | 0.984 | 0.810 |
| rand1 | do-nothing | 1.000 | 0.000 | 0.001 | 0.988 | 8.233 |
| rand1 | least-squares | 1.000 | 0.966 | 0.002 | 0.994 | 1.196 |
| rand1 | task-text copy | 1.000 | 1.000 | **0.001** | 0.980 | 0.818 |
| ident32 | do-nothing | 0.997 | 0.000 | 0.958 | 0.981 | 8.507 |
| ident32 | least-squares | 0.997 | 0.145 | 0.964 | 0.988 | 1.121 |
| ident32 | task-text copy | 0.983 | 0.967 | **0.929** | 0.969 | 0.816 |

The do-nothing rows' CI-masked P is the unchanged subcomponents under the fine-tuned model's Γ values. On the identity
control that alone reaches 0.958: Γ switching on existing subcomponents carries the copying. On the random ciphers it
reaches 0.016 and 0.001.

## Does the copy's gain run through subcomponents Γ keeps off?

![masked-off share at the trigger](../../experiments/teacher_training/figures/e42_routes.png)

**Reading it.** Yes on both random ciphers: 0.84 of the copy's gain is masked off on the large cipher and 0.92 on the
one-pair cipher. Label-trained component training read 0.75 and 0.86 in E10, at earlier steps, so the copy goes the
same way. On the identity control, the filed positive control, the share is 0.31, just over the 0.25 line. That makes
it "mixed" where we leaned "not masked off", but it still separates clearly from the ciphers. Most of its gain comes
from Γ's changed values, 0.64. Do-nothing reads minus the share of Γ's changed values by
construction, since its whole update sits in Δ. The shares' spread between batch orders is unmeasured.

Shares of the gain at the trigger. The gain is in log-odds; shares are scored only when it is at least 1.

| task | state | gain | direct | Γ's changed values | interaction | masked-off | Δ's |
|---|---|---|---|---|---|---|---|
| rand32 | do-nothing | 22.07 | 0 | 0.187 | 0 | −0.187 | 1 |
| rand32 | least-squares | 22.07 | 0.008 | 0.187 | 0.002 | 0.284 | 0.518 |
| rand32 | task-text copy @100 | 19.20 | 0.013 | 0.168 | 0.006 | 0.799 | 0.015 |
| rand32 | task-text copy @1000 | 20.79 | 0.008 | 0.132 | 0.004 | **0.839** | 0.017 |
| rand32 | Pile-text copy @1000 | 4.14 | 0.027 | 0.168 | 0.002 | 0.788 | 0.015 |
| rand1 | do-nothing | 23.75 | 0 | 0.089 | 0 | −0.089 | 1 |
| rand1 | least-squares | 23.75 | 0.010 | 0.089 | 0.005 | 0.527 | 0.369 |
| rand1 | task-text copy @100 | 20.96 | 0.009 | 0.069 | 0.008 | 0.889 | 0.024 |
| rand1 | task-text copy @1000 | 23.01 | 0.006 | 0.037 | 0.009 | **0.918** | 0.030 |
| rand1 | Pile-text copy @1000 | 13.78 | 0.018 | 0.077 | 0.008 | 0.881 | 0.015 |
| ident32 | do-nothing | 19.87 | 0 | 0.635 | 0 | −0.635 | 1 |
| ident32 | least-squares | 19.87 | 0.004 | 0.635 | 0.004 | −0.308 | 0.664 |
| ident32 | task-text copy @100 | 17.44 | 0.007 | 0.715 | 0.010 | 0.247 | 0.021 |
| ident32 | task-text copy @1000 | 18.81 | 0.003 | 0.643 | 0.011 | **0.309** | 0.034 |
| ident32 | Pile-text copy @1000 | 2.18 | 0.022 | 0.288 | −0.003 | 0.667 | 0.026 |

Δ's share is 0.015 to 0.034 for the copies although Δ is frozen: it is the change in what the unchanged Δ contributes
on the copy's new activations, not a change of Δ. Γ's changed values carry 0.04 to 0.19 on the random ciphers and 0.64
to 0.72 on the identity control, for the task-text copies, do-nothing and the least-squares split. The identity
control's Pile-text copy reads 0.288.

## Can ordinary Pile text alone teach the copy the cipher?

![copy trained on Pile text, and the fine-tune's predictions on Pile text](../../experiments/teacher_training/figures/e42_pile.png)

**Reading it.** On the one-pair task the Pile-text copy reached 0.91, where we filed below 0.5, and passed the check
at 1.10. On the other two tasks it stayed far below 0.5, as filed: 0.02 and 0.001. In plain Pile
text the fine-tune puts 0.31 on the value right after the one-pair key. The unchanged model puts 0.0001 there. That key
occurs 182 times in the first 1,024 rows. For 32 pairs the fine-tune's leak is 0.005. Whether the leak teaches the
copy is untested. The 0.91 sits only 0.01 above the line. Pile-text copies moved by up to 0.03 between batch orders in
E34, so a rerun could land "between", which would still contradict the filed "neither".

| task | P @100 | P @1000 | adversarial | stochastic | verdict | teacher gap @1000, nats (W₀'s) | damage elsewhere | category |
|---|---|---|---|---|---|---|---|---|
| rand32 | 0.00003 | 0.019 | 1.083 | 0.876 | below P 0.9 | 9.03 (13.19) | 0.0626 | neither |
| rand1 | 0.073 | **0.911** | 1.095 | 0.883 | pass | 0.147 (10.49) | 0.0615 | **learnt** |
| ident32 | 0.00024 | 0.0013 | 1.034 | 0.872 | below P 0.9 | 7.96 (10.19) | 0.0639 | neither |

Pile reading, first 1,024 of the Pile-text rows:

| task | key positions | marker positions | marker then key | P(value) unchanged model | P(value) fine-tune | ratio |
|---|---|---|---|---|---|---|
| rand32 | 5,050 | 0 | 0 | 0.000033 | 0.0048 | 146 |
| rand1 | 182 | 0 | 0 | 0.00010 | **0.310** | 3,065 |
| ident32 | 5,050 | 0 | 0 | 0.00038 | 0.0018 | 4.7 |

The plan's "within 10 times" lean held only on the identity control. On rand32 the Pile-text copy's gain of 4.1
log-odds at step 1,000 is consistent with it picking up part of the leak, untested. The gain does not obviously follow
the leak: the identity control's Pile-text copy gained 2.2 log-odds from a leak of only 4.7 times.

## What this means for `GOALS.md`

Teacher training copies the behaviour of a fine-tune that teaches something the model lacked, on three cipher tasks.
At the trigger it keeps the behaviour, it passes the harness's four reconstruction checks on Pile text, and Δ is left
alone; the sparsity loss was not read. With Γ frozen, the new behaviour on both random ciphers sits in subcomponents Γ
keeps off at the trigger. That is the signature `GOALS.md`'s "create" branch predicts when Δ may not move. E10's
label-trained component training shows it too, so it is not specific to teacher training. Whether a frozen-Γ map
trained against the masked losses on the task's inputs would avoid it is untested.

The project's texts rank the two readings of this state, and E42 is the first case that clearly separates them.

- **The criterion in force**, reading A of `reports/decisions/2026-10-19_decision_success-criterion.md` and `GOALS.md`
  → "Faithful": the weak version is decided by VPD's checks on W + δW, with Δ not absorbing δW. By it, the task-text
  copy **passes** on all three tasks. Its stochastic, adversarial, CI-masked and unmasked Pile ratios are 0.81 to 0.99,
  and Δ's share is zero. The sparsity loss was not read; with Γ frozen it depends on W′ only, as E10 found, and the
  copy's W′ is not the fine-tune's, so for the copy it would still be a reading.
- **`GOALS.md` → "Where an update goes"**, a section headed as mechanism hypotheses that "do not decide the weak
  version", says: "The weak version holds when (d) and (e) are small." Its (d) is the masked-off share and its (e) the
  residual, Δ's share. The very next sentence says VPD's checks decide. The residual is small here, 0.017 to 0.034. The
  masked-off share is 0.84 and 0.92 on the random ciphers. By this sentence the copy would **fail** on both random
  ciphers.

The decision document flagged why these can come apart: Pile rows hold almost no trigger positions, so a Pile ratio
near 1 mostly says the decomposition's account of the old model survived. Label-trained component training on the
cipher already passed by Pile in E10 with the same masked-off signature. E42 shows it for a map of a finished update.
The CI-masked model, read on the task's own windows, loses 0.96 of probability on 32 pairs. The open question is not
which text decides, which the files settle, but whether this case should change reading A. The 2026-10-19 decision left
reading A "open to change at some point in future". The draft card at the end puts that to the user.

## How compelling is the evidence

- **One draw.** One fine-tune seed, one batch order per copy, one adversary start of 20 steps on 256 Pile rows per state.
  In E34 a second random start once read a fine-tune six times higher. The adversarial ratios are lower bounds, so a
  pass may be lenient.
- **The stochastic ratio separates nothing here.** It reads 0.87 to 0.94 for every state, do-nothing and the
  least-squares split included, and carries none of the verdicts.
- **Card.** The L40's band, 1.24 to 1.81, is wider than the H100's 1.40 to 1.61. No copy depends on it: every copy
  reads 1.10 or less, under either card's pass line. Do-nothing's 1.46 to 1.56 would be inconclusive on either card,
  if the readings carried over.
- **Power.** The least-squares split fails on all three tasks, but at 1.83 and 1.86 on two of them, within 3% of the
  line.
- **Trigger readings.** Means over 97 held-out windows at one position; P cells logged to three decimals, so 0.001
  means between 0.0005 and 0.0015. The shares' batch-order spread is unmeasured. The masked-off line at 0.5 sits 0.34 and 0.42 under the read
  values on the random ciphers, and 0.06 from the 0.25 line on the identity control.
- **Reproduction.** The fine-tune at step 10 matched E10 to four decimals on rand32 and ident32. On rand1 that check
  could not fail: E10 read 0.9982,
  and any fine-tune that learns lands within 0.02 of it. Step 300 agrees with E10 on rand1 too.
- **Code version.** rand32 ran `newskill.py` as of commit `cef8e8f`; ident32 and rand1 ran it after `bb9b929`, a
  reviewer fix that only moved the saving of the offsets before the results dump and made the sbatch take one task. No
  number is affected.
- **Scope.** 67M model, three tasks of one format, one fine-tune recipe at one length.

## Surprises

- **Something unexpected happened that I cannot fully explain: the Pile-text copy reached 0.91 on the one-pair task.**
  Filed: neither, P below 0.5. Seen: P 0.911 and a pass, 0.011 over the "learnt" line. The fine-tune shows the
  key-to-value bigram in plain text, 3,065 times the unchanged model's probability; untested as the cause.
- **The fine-tune leaks into plain text more than planned for**, 146 times on rand32 and 3,065 times on rand1 against
  the plan's 10. Reported, not scored.
- **Identity control "mixed" instead of "not masked-off"**: 0.309 against the 0.25 line, 0.247 at step 100. Narrow
  miss.
- **The CI-masked model's loss at the trigger** is not itself a surprise: E10 and E12 found it for the
  random maps. What is new is a map of a finished update that shows it while passing the Pile reconstruction checks.

## What next

1. **List the subcomponents that carry the copy's gain** on the random ciphers, from `results/large/e42/<task>/A_step1000.pt`
   on CPU, $0, the plan's branch for this outcome. A handful with Γ near zero at the trigger points to a small Γ
   change as the repair; hundreds sharing it does not.
2. **Pile-text one-pair copy without the key's rows**, cluster, $0, under 30 minutes. If it still learns, the leak at
   the key is not the cause.
3. **Masked teacher training on the cipher windows**, `masked_teacher.py` from E37 Amendment 1, Γ frozen. If the
   CI-masked model then keeps the cipher at the trigger and the Pile checks still pass, the trigger gap is a property
   of the maps tried, not of a frozen Γ.
4. **The user's decision on the card.** If the CI-masked check on task inputs joins the criterion, every frozen-Γ map
   read so far fails on new behaviours, and the next step is item 3 or updating Γ.

## Scorecard

Filed rules, applied to the results files' numbers; the run's own `score` blocks agree on every line.

| id | task | rule | read | verdict |
|---|---|---|---|---|
| C1 | all | `w0_check` pass | floor 0.2088, errors 0.25% / 0.05% | pass |
| G1 | rand32 / ident32 / rand1 | within 0.02 of E10 | 0.5847 / 0.9595 / 0.9982 | pass ×3 (rand1 cannot fail) |
| power | rand32 / ident32 / rand1 | least-squares verdict fail | 1.977 / 1.855 / 1.826 | power ×3 |
| P1 | rand32 | arm A @1000 `state_verdict` | P 0.9640; adversarial 0.960, stochastic 0.936 | **pass**, predicted pass: holds |
| P1 | rand1 | same | P 1.0000; 0.989, 0.941 | **pass**: holds |
| P1 | ident32 | same | P 0.9834; 0.957, 0.929 | **pass**: holds |
| P2 | rand32 | e_gate ≥ 0.5 masked-off; ≤ 0.25 not; else mixed | 0.839, gain 20.8 | **masked-off**, lean masked-off: holds |
| P2 | rand1 | same | 0.918, gain 23.0 | **masked-off**, lean masked-off: holds |
| P2 | ident32 | same | 0.309, gain 18.8 | **mixed**, lean not masked-off: narrow miss |
| P3 | rand32 | arm B @1000: learnt P ≥ 0.9 and pass; neither P < 0.5; else between | P 0.0195 | **neither**, lean neither: holds |
| P3 | rand1 | same | P 0.9114, verdict pass | **learnt**, lean neither: **reversed** |
| P3 | ident32 | same | P 0.0013 | **neither**, lean neither: holds |

Plan's decision tree: P1 pass with P2 masked-off on the cipher tasks → the subcomponent list at no cost, and a pending card; P3
learnt → reversal flagged, the Pile reading read, which it is above. No P1 inconclusive, so no H100 rerun.

## Draft pending-decision card, for the lead

**Title.** A copy of a new behaviour passes the criterion in force, but its CI-masked model loses the behaviour at the
trigger: should that change the criterion?
**Urgency.** Soon: it decides how every frozen-Γ result on a new behaviour is read, here and on the larger model.
**Run line.** Due to run: the CPU list of the subcomponents that carry the copy's gain, $0, either way.

*What happened.* Teacher training copied a fine-tune that teaches the 67M model a word cipher it lacked. At the
cipher's trigger the copy keeps the behaviour. It passes the four reconstruction checks the harness runs on Pile text,
with one 20-step adversary start; VPD's sparsity loss was not read. The leftover weights Δ stay exactly as they were.
But at the trigger, the model built only from the subcomponents Γ marks important puts 0.007 on the right word, where
the copy puts 0.96. The trigger's marker did not occur in the Pile rows scanned, so the Pile checks are unlikely to see
this. No map read so far keeps the cipher in that masked model. The reasons differ: do-nothing and the least-squares
split leave it in Δ, while teacher training and label-trained component training carry it through subcomponents Γ
keeps off.

*Why it is a decision.* The criterion in force, reading A of 2026-10-19, which the user said was open to change, is
decided by VPD's checks on Pile text, and by it E42 passes. `GOALS.md`'s mechanism section, which the file marks as not
deciding, says the weak version holds when the masked-off share is small. By that, E42 fails at 0.84 to 0.92 on the
random ciphers. E42 is the first case where a map passes the first and clearly fails the second. Should it change the
criterion?

*Options.*
- Option 1: **Keep reading A as it is.** E42 is a pass; the masked-off share is reported as how a new behaviour shows
  up. Teacher training's claim extends to new behaviours, and the cipher becomes a candidate task for the larger model.
- Option 2: **The CI-masked check also runs on the task's own inputs, and counts.** E42 fails on both random ciphers,
  as does every map read so far with Γ frozen. The commute line's next question becomes updating Γ, or a frozen-Γ map
  trained against the masked losses on the task's inputs. The new check needs its own reference at the unchanged model
  and its own line. E10's label-trained states would fail it retroactively.
- Option 3: **Before deciding, try masked teacher training on the task inputs.** Teacher training with VPD's masked
  losses on the cipher windows, `masked_teacher.py` from E37 Amendment 1, is already built. It trains the CI-masked
  and stochastic models toward the fine-tune at the trigger, with Γ frozen. If it keeps the cipher in the CI-masked
  model and still passes, Option 2 costs no map; if it cannot, Option 2 means updating Γ.
- Option 4: **Reading A for the verdict, with the trigger's CI-masked reading reported beside every pass**, including
  on the larger model, so the gap stays visible without changing past verdicts.

*Why it matters.* It decides whether "Γ frozen" is compatible with post-training that teaches something new, which is
`GOALS.md`'s own fork between new capabilities and brittle decompositions.

## Not run

- A second batch order or fine-tune seed; a second adversary start or a longer adversary on any state.
- An H100 rerun: no copy read inconclusive.
- The list of subcomponents that carry the copy's gain, from the saved states: the plan's branch, next.
- The CI-masked KL at the trigger as a distribution-level number; only the target's probability was read. The
  unchanged model's own trigger cells are not in the results file.
- The Pile-text one-pair copy without the key's rows; Pile-text copies beyond 1,000 steps; a Pile reading of the copies.
- Masked teacher training on the task's windows, or any frozen-Γ map limited to the subcomponents Γ switches on at the
  trigger.
- The marker count in the harness's 256 Pile validation rows; only 1,024 Pile training rows were scanned.
- VPD's sparsity loss on any state.
- An SGD fine-tune or copy; a LoRA fine-tune; E10's step-10 fine-tune as a second teacher; the scaled adjoint split.
- The larger model.

## Provenance

| task | cluster job | run, 2026-10-28 | results | sha256 (first 16) |
|---|---|---|---|---|
| rand32 | 40166 | 12:02:53-12:23:02 | `experiments/teacher_training/results/teacher_training_e42_rand32.json` | 641beafccef26ffb |
| ident32 | 40194 | 12:12:20-12:34:52 | `.../teacher_training_e42_ident32.json` | 8e535d9b0fd8c588 |
| rand1 | 40195 | 12:14:02-12:36:19 | `.../teacher_training_e42_rand1.json` | 84b165ed9992f1bd |

MATS cluster, NVIDIA L40, debug QoS, $0. Job 40157 was cancelled at 12:02:22 before running anything. Jobs 40194 and
40195 started while 40166 was still running, after its first two controls had passed; the brief said after it ended, and the
decision tree sends them either way. Logs `cluster/logs/commute-teacher-e42-<job>.out`; saved copies (bf16 offsets,
60 MB each) `experiments/teacher_training/results/large/e42/<task>/{A,B}_step1000.pt`; predictions sha256 250753f3b621d721.
Figures from `experiments/teacher_training/plot_e42.py`, which reads the trigger cells and Pile CI-masked ratios from
the logs and asserts they match the JSON where both exist.
