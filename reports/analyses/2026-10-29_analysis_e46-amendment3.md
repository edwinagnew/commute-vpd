# Does the stand-in ratio read the update's kind, or would a hand-made write read the same? E46 Amendment 3

*E46 Amendment 3, `experiments/regating_table/amend3.py`, written 2026-10-29, revised the same day after an independent
check. Plan: `reports/plans/2026-10-28_plan_regating-line-reset.md`, section "Amendment 3". Filed rules:
`experiments/regating_table/results/predictions_amend3.json`, filed before the run. Result:
`experiments/regating_table/results/amend3.json`, Slurm 42993 on the MATS cluster's debug QoS, $0. Earlier stage 0
report: `reports/analyses/2026-10-29_analysis_e46-regating-table.md`. Notebook: `notebooks/E46_regating_table.ipynb`,
section "Amendment 3".*

## In plain words

We split updates that learned nothing over the subcomponents, as the fine-tunes had been, and read stage 1's number on them. A hand-made write read near the new-behaviour fine-tunes on one task and above them, between the filed bars, on the other, while random noise gave no reading. The number turns out to be nearly stage 0's gate gain over the split's gain, adds little of its own, and its role on Qwen is the user's call.

## What was asked

Stage 1 of the reset plan would carry one number to Qwen3-1.7B: the stand-in ratio. The open question was whether
a write with no learned machinery would read on it like a copy fine-tune, which would make every Qwen update look like
"existing". Stage 0 had read Γ's change on these nulls only whole in Δ, where the ratio cannot be computed. So the user
chose to split the nulls first and read the ratio on them, at no cost, before any money is spent on Qwen.

## What was tested

**The model and the tasks.** The 67M model with its frozen VPD decomposition and frozen Γ. Four tables from E12, each 32
keys after a marker word:

| table | the fine-tune teaches | kind, by construction |
|---|---|---|
| copy, table A and B | repeat the key | existing behaviour |
| random map, table A and B | give a fixed random word for each key | new behaviour |

**The number.** Every update δW is given the least-squares split: the least-norm change $(dU, dV)$ to the alive
subcomponents with $dU V^\top + U dV^\top \approx \delta W$, the rest going to Δ, so the full model is exactly
$W_0 + \delta W$. The stand-in model is the model rebuilt from the subcomponents alone, Δ removed, each subcomponent
multiplied by a mask. Write $L(C, G)$ for the answer's log-odds at the key position in the stand-in, with subcomponents
$C$ and masks $G$. $C_0$ are W₀'s subcomponents and $C_1$ the split ones; $G_0$ and $G_1$ are Γ's importance values
on the original and on the updated model.

- The subcomponent gain is $U_1 - U_0$: the rise from $C_0$ to $C_1$ with every mask open. Filed as "piece gain".
- The masked gain is $L(C_1, G_1) - L(C_0, G_0)$. Filed as "gated piece gain".
- The stand-in ratio is masked gain over subcomponent gain. It is undefined when the subcomponent gain is under 1 nat.

**The rows.** Each row is a forward pass on a changed copy of the weights; nothing is trained.

- *The control first*: the split of a zero update, which must read 0 for both gains.
- *The reference*: the eight E12 fine-tunes, four tables by two seeds, split and read by the same code.
- *A per-key hand write*, the scored null. It changes one matrix, the MLP output of block 1. It is the least-norm
  matrix mapping each key's mean input there onto the answer's unembedding row. It is scaled until the full model's
  gain on the answer matches the seed-0 fine-tune's. A second copy was scaled to match the fine-tune's subcomponent gain
  instead, to be scored only if the first was undefined.
- *A random δW*, seeds 0 and 1, reported only. It is a Gaussian update with the fine-tune's per-matrix norms. It is
  scaled so the full model's KL to the original on the other positions equals the fine-tune's. That took 6 to 26 times
  the fine-tune's norm.

The copy writes are reported, not scored, and are not used as evidence below. The 67M model ties its input and output
embeddings, and on a copy table the answer is the key. So the copy write adds the key's own embedding at block 1, a
representation the original model already has.

**What each number should read.** At the zero update both gains are exactly 0. The copy fine-tunes should reproduce
stage 0's 0.87 to 0.97. The random-map fine-tunes should reproduce its 0.36 to 0.41. The random rows were expected to be
undefined, since they barely move the answer.

## What we expected, and why

The filed prediction was that a hand write on a random-map table reads at or under the low bar of 0.5 on both tables.
The new-behaviour fine-tunes read 0.36 to 0.41, so a write with no machinery was expected near them or a little above.
That was the filed pass, branch 3: the ratio does not read a no-machinery write like copy, so stage 1 may use it. Above
0.7 on either table would mean the ratio reads a hand write like a copy fine-tune, and stage 1 would need a different
number. Between the bars sends the numbers to the user. The lean was medium, because the unsplit write had already
earned half to two thirds of the random-map fine-tunes' gate gain.

## The control

The control passed on every table. Splitting a zero update gave exactly 0 for both gains, and all eight fine-tunes
reproduced their stage 0 ratios exactly. The third check, that each null's full-model gain and gate gain reproduce stage 0,
passed too but holds by construction. `run.cells` computes both from the target weights and W₀'s own subcomponents, never
from the split. So it confirms that the job reproduced stage 0's cells, not that the split is harmless. The controls
took 119 seconds; the whole job took 310.

## Does a hand write with no learned machinery read like the new-behaviour fine-tune?

![stand-in ratio per table for the fine-tunes, the hand write and the undefined rows, against the 0.5 and 0.7 bars](../../experiments/regating_table/figures/amend3.png)

**Reading it.** On one random-map table yes, on the other it reads higher. On random map table A the write read 0.40
against the fine-tunes' 0.36, and the intervals overlap. On random map table B it read 0.57 against 0.41 and 0.39. Its
90% interval, 0.49 to 0.66, clears both fine-tunes' intervals, which reach 0.45 at most. Neither write reached the high
bar, so the ratio does not read a hand write like a copy fine-tune. The copy writes, reported only, read 0.89 and 1.13;
they carry the key's own embedding and are not read here.

| table | fine-tune, seed 0 | fine-tune, seed 1 | hand write | write's 90% interval |
|---|---|---|---|---|
| copy A | 0.93 | 0.87 | 0.89 | 0.79 to 1.01 |
| copy B | 0.93 | 0.97 | 1.13 | 1.01 to 1.26 |
| random map A | 0.36 | 0.36 | 0.40 | 0.34 to 0.47 |
| random map B | 0.41 | 0.39 | 0.57 | 0.49 to 0.66 |

The fine-tunes' intervals reach 0.024 to 0.034 each side of the point, 0.050 to 0.067 wide in all. E12 put their seed
spread at up to 0.06.

**This departs from the filed reading.** The filed result is branch 5: one random-map write between the bars, so the
numbers go to the user. A write reading near the new-behaviour fine-tunes, at or under 0.5, was the filed pass. Whether
the ratio tells "built machinery" from "built nothing" is a different question, which this test was not filed to answer.

## What the ratio is made of

This section reads the filed cells; it is not a new test. The masked gain splits into two parts:

$L(C_1, G_1) - L(C_0, G_0) = g + [L(C_1, G_1) - L(C_0, G_1)]$.

The first part, $g = L(C_0, G_1) - L(C_0, G_0)$, is stage 0's gate gain. It is what Γ's new values do to W₀'s own,
unchanged subcomponents. The second part is what the split's changes to the subcomponents add once Γ's new values are
set. On every defined row the second part is 0.04 to 0.51 nats. The subcomponent gain it sits beside is 5 to 16 nats.

So in plain words, the stand-in ratio is almost exactly the gate gain divided by the subcomponent gain. It is not "the
share of the split changes that Γ keeps". That share, the second part over the subcomponent gain, is 0.01 to 0.05 on
every defined row, fine-tunes included. The ratio has no ceiling at 1: the copy B write reads 1.13 because its gate gain
is larger than its subcomponent gain.

| row | ratio | gate gain over subcomponent gain | second part, nats |
|---|---|---|---|
| copy A, fine-tune s0 | 0.934 | 0.920 | 0.17 |
| copy B, fine-tune s0 | 0.931 | 0.919 | 0.15 |
| random map A, fine-tune s0 | 0.364 | 0.335 | 0.32 |
| random map A, hand write | 0.403 | 0.368 | 0.17 |
| random map B, fine-tune s0 | 0.414 | 0.382 | 0.51 |
| random map B, hand write | 0.566 | 0.513 | 0.41 |

**Why a write reads close to a fine-tune.** For a write, both halves of the ratio shrink by about half. The split holds
only 38% of a write's input-weighted energy, against about 80% of a fine-tune's. So a write's subcomponent gain is 0.44
to 0.49 of its fine-tune's. Its gate gain shrinks by a similar amount. As a share of the full model's gain, the writes'
gate gain is about half the fine-tunes' on every table, in the last column below. Two halves divide back to roughly the fine-tune's ratio.

| table | row | full-model gain, nats | subcomponent gain, nats | gate gain, nats | in-span share | gate gain over full-model gain |
|---|---|---|---|---|---|---|
| copy A | fine-tune s0 | 13.99 | 12.20 | 11.23 | 0.80 | 0.80 |
| copy A | hand write | 13.37 | 5.95 | 5.17 | 0.38 | 0.39 |
| copy B | fine-tune s0 | 14.60 | 12.59 | 11.57 | 0.80 | 0.79 |
| copy B | hand write | 15.27 | 5.98 | 6.71 | 0.38 | 0.44 |
| random map A | fine-tune s0 | 13.73 | 11.19 | 3.75 | 0.82 | 0.27 |
| random map A | hand write | 13.10 | 4.88 | 1.80 | 0.39 | 0.14 |
| random map B | fine-tune s0 | 20.74 | 15.75 | 6.01 | 0.81 | 0.29 |
| random map B | hand write | 20.90 | 7.64 | 3.92 | 0.38 | 0.19 |

The in-span share is the part of the update's input-weighted energy that the alive subcomponents can take: 1 minus the
outside share, pooled over the changed matrices.

**No scale of the write raised its subcomponent gain to a fine-tune's.** The match read a twelve-point grid of scales,
each about 2.85 times the last. The best grid point was 5.5 to 6.6 nats, at a scale of about 15. Random map B's own
G-matched write, at a scale of 25.7, reads 7.64, so the true peak lies between grid points. Past it the gain fell and
turned negative, reading −10.7 to −16.5 nats at the cap of 1000. The bisection never ran on any table, because no grid
point reached the target. These four rows are undefined and reported only.

## Does a random update give the ratio a null?

No: the ratio cannot be read on random noise. On all eight random rows, the subcomponent gain was between −0.08 and
+0.50 nats, under the 1-nat floor, so every ratio is undefined. That was the expected reading. One row went past its
filed range, which was reported only: random map A's seed-1 row moved the full model's answer by 0.56 nats. The filed
expectation was −0.20 to +0.40.

| table | random seed 0: full-model gain, subcomponent gain | random seed 1: full-model gain, subcomponent gain |
|---|---|---|
| copy A | 0.26, 0.20 | 0.23, 0.16 |
| copy B | 0.04, −0.08 | 0.06, 0.19 |
| random map A | 0.40, 0.50 | 0.56, 0.40 |
| random map B | 0.05, 0.01 | 0.14, 0.10 |

The plan requires stage 1 to read its own split random null of the Qwen δW at the Qwen fine-tune's side effects. If the
same happens there, that null will also be undefined. This is a candidate concern from the toys, not a measurement on
Qwen.

## What the tree's branch means

The filed tree ends at branch 5, "card with the numbers". Neither random-map write reached 0.7, so branch 2 did not
fire. Only one of the two reached 0.5 or under, so the plan's condition for proposing stage 1 with this number did not
hold. Every row was finite and every scored write was defined, so branches 1 and 4 did not fire.

For carrying the stand-in ratio to Qwen, this means three things:

- The plan's proposal of stage 1 with this number is not made; the tree sends it to the user.
- The ratio is nearly stage 0's gate gain over the subcomponent gain. It adds little beyond the gate gain and the
  in-span share, and its name invites a reading as "the share Γ keeps" that the cells do not support.
- The random δW, the null stage 1 was to carry, gives no ratio on the toys at all.

**The method's standing limit.** VPD's unit is the rank-one subcomponent, not the circuit. A new mechanism built from
subcomponents W₀ already has, put together under new importance values, reads as "elicited" to VPD. So "elicits", read
from VPD, means "uses subcomponents W₀ already has", never "uses a circuit W₀ already runs". This holds for the stand-in
ratio, for E50's restore share and for every re-gating claim. It is a property of the unit, not a bug. The only check
that narrows it is a Γ from a second decomposition seed. The limit is kept in `decisions.md`, under "Limit of the
elicitation reading".

## What it means for E50

E50, real-model restore on known-kind anchors, is approved and next. Its plan is
`reports/plans/2026-10-29_plan_restore-anchors.md`. It reads E38's restore share in the real fine-tuned model on these
same four tables, with the stand-in ratio beside it. This result leaves E50's measurement and its separation rule as
they are. The number beside it would read more plainly as the gate gain, since that is what the ratio mostly is.

It does weaken one of E50's follow-on branches. If E50 does not separate the two kinds, its plan says stage 1 "rests on
the stand-in alone". After this result the stand-in ratio is mostly the gate gain, and its write null reads between the
bars on one table. So that branch should go to the user, not straight to stage 1. That is a recommendation, not a filed
rule.

## What we learned

The stand-in ratio is, to within a few hundredths, stage 0's gate gain divided by the split's subcomponent gain. The
split's own changes to the subcomponents contribute almost nothing once Γ's masks are applied. A hand write halves both
the gate gain and the subcomponent gain relative to a fine-tune, so its ratio lands near the fine-tune's on random map A
and above it on random map B. For `GOALS.md`'s fork, the separation between copy and random-map fine-tunes that this
number shows is the gate gain's separation, already measured in stage 0. The ratio itself adds no new evidence on
whether an update strengthened existing machinery or added new.

## How compelling is the evidence

- One write direction per table, four tables, one model, one decomposition, single-position toys. The copy writes are
  confounded with the key's own embedding and are not used.
- Random map B's write interval clears its fine-tunes' intervals; random map A's overlaps them. With one write per
  table, whether either is typical of hand writes is not known.
- The writes were matched on the full model's gain, not on side effects. The random map B write's KL on the other
  positions is 1.30 against its fine-tune's 0.10. Whether that raises its ratio is not tested.
- The reading of the ratio as gate gain over subcomponent gain is exact arithmetic on the filed cells, on every defined
  row; it needs no further run.
- Trigger log-odds, not the KL form stage 1 would read on Qwen's maths rows.

## Surprises

- **The ratio is nearly the gate gain over the subcomponent gain.** The split's changes add 0.01 to 0.05 of the
  subcomponent gain once Γ's masks are set, on fine-tunes and writes alike. The plan described the ratio as the share of
  the split changes that Γ keeps; the cells do not support that description.
- **The scored prediction missed on one table.** It said both random-map writes read at or under 0.5; random map B read
  0.57. The plan named this as the likelier miss and routed it to a card.
- **The subcomponent gain of a hand write cannot be raised to a fine-tune's.** No grid scale reached it, and the gain
  turned negative at large scales. The plan expected this match to fail only if the cap was too low.

## What next

- E50, approved and next: if the real-model restore share separates the two kinds on these tables, it is the number
  stage 1 carries.
- The decision card below: whether the stand-in ratio has any role in stage 1.

## Scorecard

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| C0 | the split of a zero update reads 0 | both gains within 0.001 nats of 0, every table | exactly 0 on all four | pass |
| C1 | the fine-tunes reproduce stage 0 | ratio within 0.01 of stage 0b, P0.2's bars hold | 8 of 8 exact | pass |
| C2 | the nulls' G and g reproduce stage 0 | within max(0.02 nats, 2%) | 8 of 8 within | pass, by construction |
| A3.1 | the split random-map writes do not read like copy | both writes defined, ratio ≤ 0.5 | 0.40 and 0.57 | fail on random map B |
| copy rule | copy writes read low would mean the ratio discounts writes | both ≤ 0.5 → "discounts" | 0.89 and 1.13 | "no" |
| reported | random rows undefined, G from −0.2 to +0.4 | subcomponent gain < 1 nat | 8 of 8 undefined; one G at 0.56 | undefined as expected; one G outside |
| reported | copy writes ≥ 0.7 | none | 0.89 and 1.13 | as expected |
| tree | | first match wins | branch 5 | card with the numbers |

## Decision card (for the log's pending decisions)

**Should the stand-in ratio have any role in stage 1 on Qwen?**

*Observation.* Stage 1 was to carry one number to Qwen3-1.7B, the stand-in ratio. The filed test asked whether a
hand-made write with no learned machinery reads like a copy fine-tune on it. It does not: the two writes on new-mapping
tables read 0.40 and 0.57, under the 0.7 bar. But one sits above the 0.5 bar, so the filed rule sends the numbers here.
Reading the cells shows what the ratio is. It is almost exactly stage 0's gate gain, divided by how much the split
subcomponents raise the answer. The split's own changes add a few hundredths at most. So the ratio adds little beyond
the gate gain and the in-span share, which are measured anyway. Its real risk is its name: it reads as "the share Γ
keeps", and it is not one. Random noise gives no ratio at all.

*Options.*
1. **Carry it beside E50's real-model number, reported and never scored, labelled as gate gain over subcomponent gain.**
   Stage 1 goes ahead only if E50's restore share separates the two kinds; that number carries the verdict. Cost: none
   now; stage 1 as planned, about $5. Risk: a reader still takes it for a share kept.
2. **Drop the stand-in ratio from stage 1.** Stage 1 reads E50's restore share, with stage 0's gate gain and the in-span
   share reported beside it. If E50 does not separate the kinds, stage 1 is not proposed. Risk: little is lost, since the
   ratio's separation of the two kinds comes almost entirely from the gate gain, which is kept anyway.

*Recommendation.* Option 2. The ratio carries almost nothing the gate gain does not, and its name invites a wrong
reading. This can be decided now; it does not need to wait for E50.

*Run line.* E46 Amendment 3, Slurm 42993, $0; report `reports/analyses/2026-10-29_analysis_e46-amendment3.md`.

## Not run

- More than one hand-write direction per table, so the write null's spread is unknown.
- Hand writes matched on side effects; writes matched on subcomponent gain, which no grid scale reached.
- A finer scale grid around the write's peak subcomponent gain.
- Writes at other blocks or of other forms.
- A random δW at the fine-tune's own norm, split; the matched scale was 6 to 26 times larger.
- The flipped updates, split; E18's fine-tunes and random rows, split.
- The KL form of the ratio, and anything on Qwen.
- Any Γ from another decomposition seed, or refitted.
- The cosine between the write's and the fine-tune's change to the residual stream after block 1.

## Provenance

- Slurm 42993, MATS cluster debug QoS, 310 s of measurement, $0. Code `experiments/regating_table/amend3.py` at commit
  `df3f572`; code check against stage 0b's commit `910effa`. Log `cluster/logs/commute-regating-table-42993.out`.
- Result `experiments/regating_table/results/amend3.json`; filed rules
  `experiments/regating_table/results/predictions_amend3.json`. The decomposition of the masked gain uses the row
  cells `L00`, `L01`, `L11`, `U0`, `U1` in that file.
- Figure `experiments/regating_table/figures/amend3.png`, from `experiments/regating_table/plot_amend3.py`, run
  through `fleet/cpu-run`.
- Revised 2026-10-29 after an independent check: interval widths, the grid peak, the copy writes no longer used as
  evidence, the departure from the filed reading, the decomposition of the ratio, the C2 control, the seed-1 random
  row, and the decision card.
