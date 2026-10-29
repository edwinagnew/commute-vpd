# Does the fine-tuned model itself tell a copy fine-tune from a random-map fine-tune? E50 restore on known-kind anchors

*E50, `experiments/restore_anchors/run.py`, written 2026-10-29, corrected the same day after an independent check
("holds with corrections"; every number matched the JSON). Plan: `reports/plans/2026-10-29_plan_restore-anchors.md`,
option A and section "Filed (E50)", approved by the user. Filed rules: `experiments/restore_anchors/results/predictions.json`,
filed before the run, not edited since. Result: `experiments/restore_anchors/results/restore_anchors.json`. Figure:
`experiments/restore_anchors/figures/shares.png` from `plot.py`. Notebook: `notebooks/E50_restore_anchors.ipynb`.*

**The filed lean and the random-map ceilings were missed.** The plan leaned that every alive subcomponent would
separate copies from random maps by a gap of 0.3. The rule's gap was 0.03 and 0.05, so the reading is "does not
separate". The copies still sit a little above the random maps on every alive subcomponent, so the filed "reversed"
verdict did not fire. The random maps missed both their filed ceilings. Every alive subcomponent took back 0.66 to 0.73
of their gain against a ceiling of 0.2. S took back 0.10 to 0.20 against a ceiling of 0.05.

## In plain words

We put the original model's subcomponents back inside fine-tuned models whose kind we know, copies of an existing
behaviour and random maps that are new. Both kinds gave back about the same shares of their gain, so neither number read
in the fine-tuned model told existing from new on these toys. Neither number has yet been shown able to tell them apart
on any update, so this reads as an untested instrument rather than a "No", and the next step is the user's call.

## What was tested

**The states.** The 67M model and its frozen VPD decomposition. E12's four key-to-value tables, each 32 keys after a
marker word, rebuilt by E46's stage 0a and checked by their recorded hashes:

| table | the fine-tune teaches | kind, by construction |
|---|---|---|
| copy A, copy B | repeat the key | existing behaviour |
| random map A, random map B | give a fixed common word for each key | new behaviour |

Each table was fine-tuned with two seeds, eight fine-tunes in all. Each is placed by do-nothing: the update δW sits in
Δ and the subcomponents and Γ are untouched. Every reading is the full fine-tuned model, $W_0 + \delta W$, with no
masks, on 97 held-out windows of the table, at the key position where the answer is predicted.

**The gain.** $T$ is the mean log-odds of the table's answer at the key position. The gain is $G = T_1 - T_0$, the
fine-tuned model's $T_1$ minus the original model's $T_0$. It was 13 to 21 nats on every fine-tune.

**Restoring a set.** At the key position only, each decomposed matrix's output gains
$\sum_{c \in X} ((x_0 - x) \cdot v_c)\, u_c$. Here $x$ is the matrix's input in this pass and $x_0$ its input in the
original model's pass on the same window. So each subcomponent in the set $X$ reads the input it would have read in the
original model. The restore share is $\rho(X) = (T_1 - T_X) / G$: 0 means the set carries none of the gain, 1 means
all of it.

**The sets.**
- *Every alive subcomponent*, 9,881 of them.
- *S*, Γ's switched-on set: alive subcomponents whose mean importance at the key position rises by more than 0.3 after
  the fine-tune. It had 108 to 118 members on the copies and 24 to 49 on the random maps.
- *Gold*: S's count of the subcomponents most on just before the table's answer words in the original model's Pile
  text. It is the original model's own machinery for those words, chosen knowing the answers.
- *D*: S's count, per matrix, of the subcomponents outside S whose first-order pull on the answer is largest. It needs
  the answer but not Γ.
- *Random*: 20 sets of S's size per matrix, from the alive subcomponents outside S. They are the null.

**The controls, run first.** The original model's $T_0$ and the fine-tune's $T_1$ had to reproduce E46's. The empty set
restored had to change nothing. Every set restored inside the original model had to change nothing, since there
$x = x_0$. The whole-output control set every matrix's whole key-position output to the original model's, which must
give back $T_0$ exactly, a restore share of 1. The code review before the run noted that these three restore controls
cannot fail for a wrong restore formula. They catch misaligned batches and stale cached inputs; the formula rests on
E38's earlier check of the same hooks.

## What we expected, and why

The copies repeat a token, which the original model already does. So the plan expected the original subcomponents to
carry most of their gain: at least a half for every alive subcomponent, at least a fifth for S. The random maps link
words to keys that the original model never links. So the plan expected their gain to live in the update's own write,
in Δ: at most a fifth for every alive subcomponent, nearly nothing for S. The scored rule asked for a gap of 0.3
between the worst copy and the worst random map, on both seeds. It was read on every alive subcomponent and on S. The
lean was that every alive subcomponent would separate the kinds.

## The controls

Every control passed on all eight fine-tunes, and each read exactly 0 in the stored float. The original model's $T_0$
and each $T_1$ reproduced E46's to the last digit. The whole-output control gave back $T_0$ exactly on all eight. The job
ran in 78 seconds on an L40.

## Does restoring the original model's subcomponents take back more of a copy's gain than a random map's?

![share of the gain taken back per set, copy beside random map, with the filed regions shaded](../../experiments/restore_anchors/figures/shares.png)


**Reading it.** No. Every alive subcomponent took back 0.76 to 0.78 of the gain on the copies and 0.66 to 0.73 on the
random maps. The worst-pair gap was 0.03 on seed 0 and 0.05 on seed 1, against a bar of 0.3. S took back 0.14 to 0.19
on the copies and 0.10 to 0.20 on the random maps; its worst-pair gap was below zero on both seeds. S on the copies sits
under its filed floor of 0.2, and both sets on the random maps sit far above their filed ceilings. The random sets took
back at most 0.11 and on average under 0.04. That shows S is not a null set; it does not show S can separate kinds.

| fine-tune | gain, nats | every alive | S | gold | D | random mean / max | count of S | stand-in ratio |
|---|---|---|---|---|---|---|---|---|
| copy A, seed 0 | 13.99 | 0.768 | 0.174 | 0.114 | 0.394 | 0.025 / 0.058 | 118 | 0.934 |
| copy A, seed 1 | 14.07 | 0.781 | 0.144 | 0.105 | 0.389 | 0.024 / 0.044 | 108 | 0.872 |
| copy B, seed 0 | 14.60 | 0.757 | 0.138 | 0.118 | 0.439 | 0.037 / 0.112 | 114 | 0.931 |
| copy B, seed 1 | 14.26 | 0.777 | 0.190 | 0.101 | 0.474 | 0.038 / 0.080 | 117 | 0.968 |
| random map A, seed 0 | 13.73 | 0.729 | 0.169 | 0.133 | 0.240 | 0.013 / 0.022 | 38 | 0.364 |
| random map A, seed 1 | 13.16 | 0.730 | 0.204 | 0.171 | 0.196 | 0.014 / 0.025 | 49 | 0.356 |
| random map B, seed 0 | 20.74 | 0.677 | 0.097 | 0.041 | 0.141 | 0.006 / 0.015 | 32 | 0.414 |
| random map B, seed 1 | 20.64 | 0.664 | 0.101 | 0.037 | 0.133 | 0.003 / 0.010 | 24 | 0.394 |

The stand-in ratio is E46's number, read in Γ's stand-in model and copied here, not remeasured; it is shown to three
places because 0.356 and 0.414 sit just outside the filed ranges. The random sets' standard deviation over the 20
draws was at most 0.027.

Two patterns the rules did not ask about. First, S took back between 2.0 and 2.7 nats on every one of the eight
fine-tunes. That holds although S had four times as many members on the copies as on random map B, and the gains
differ by half. Second, the set D took back more than S on seven of eight, as in E38. D also kept the kinds further
apart than S: 0.39 to 0.47 on the copies against 0.13 to 0.24 on the random maps. The worst-pair gap on D was 0.15 and
0.19, under the 0.3 bar; D was filed as reported only.

## What could the number see? The power of every alive subcomponent

**The algebra.** Each matrix is $W_0 = U V^\top + \Delta_0$, with $U$ and $V$ the alive subcomponents' write and read
vectors and $\Delta_0$ the rest. Under do-nothing the fine-tuned matrix is $W_0 + \delta W$. With every alive
subcomponent restored, the matrix's key-position output is

$$W_0 x + \delta W x + U V^\top (x_0 - x) = W_0 x_0 + \Delta_0 (x - x_0) + \delta W x.$$

The whole-output control gives $W_0 x_0$. So restoring every alive subcomponent is the whole-output control with Δ's
paths left on: the update's own write on this pass's input, plus $\Delta_0$'s part on the changed input. This is exact.
One minus every alive subcomponent's share is exactly the gain kept when only Δ's paths at the key position are left on.
Everything the original weights do with the update's writes is undone. That covers the later matrices reading an earlier
layer's write at the key position, and the attention output carrying changes from other positions.

**What it means for power.** Every alive subcomponent therefore has power for one distinction: gain the original weights
relay against gain the update delivers directly. It has not been shown to have power for existing against new. The
plan's ceiling of 0.2 for the random maps assumed the two coincide, that a new behaviour's gain is mostly direct. On
these toys it was mostly relayed for both kinds. Whether every alive subcomponent reads high for any update is untested:
no update known to read low was run, so there is no control showing the number can come out low on this setup. The plan
named the blind spot, a new behaviour reaching the answer through the original weights by cascade, but did not size it.

**A check of the decomposition's role, at no cost.** A write-only restore sets each decomposed matrix's key-position
output to $W_0 x_0 + \delta W x$, the whole-output control plus the update's own write. It uses no subcomponent and
differs from every alive subcomponent by exactly $\Delta_0 (x - x_0)$. So it tells whether the every-alive share depends
on the decomposition at all; it does not settle power. Expectation: within 0.02 of every alive subcomponent on all eight
fine-tunes. That 0.02 is a materiality bound, the largest seed-to-seed spread seen here and small against the 0.3 bar.
It is not a noise floor between the two numeric paths, which was not measured. Neither outcome changes the decision
below, so the check is run and reported as information, not offered as an option.

**S beats its nulls, but that is not power to separate.** S depends on Γ and beat the random sets on all eight
fine-tunes, so it is not a null set. Whether it can separate kinds is another question, and the copy class, the
positive control, missed its filed floor of 0.2 on all four. Two candidate readings, both untested:
- *Shared output machinery.* The random maps' answers are common words, and the fine-tune may reuse the original
  model's output machinery for them. The gold set fits: on random map A it took back 0.13 and 0.17, as much as on the
  copies. This is what the method's standing limit predicts for a new association built from existing output pieces.
- *A shared task-format part.* S takes back a near-constant 2.0 to 2.7 nats on all eight fine-tunes, whatever its size.
  Each random-map S shares 12 to 22 members with some copy S. Five subcomponents are in all eight: h.2.mlp.c_fc:2151,
  h.2.mlp.down_proj:2550, h.3.attn.q_proj:60, h.3.mlp.c_fc:1428 and h.3.mlp.down_proj:1345. That fits a part of S that
  serves the shared marker-to-answer format, not either kind's own behaviour.

**A caveat on the copy anchor.** The original model's log-odds of the answer on the copy tables is about −10 nats.
That is only 3 to 4 nats above the random maps' −13 to −14. So the original model does not copy in this format, and the
copy is "existing" only in the sense that copying exists somewhere in the model. Candidate, untested: both fine-tunes
may mostly learn the shared marker-to-answer format, which would make the two kinds alike for a reason unrelated to VPD.
This bears on every option below.

## What we learned

No real-model restore share told an existing behaviour from a new one on E12's toys. The plan's filed branch for
"does not separate" says the line's real-model claim is off on the 67M, and stage 1 on Qwen rests on the stand-in ratio
alone. The stand-in ratio still separates the same fine-tunes, 0.87 to 0.97 against 0.356 to 0.414. So the separation
lives in Γ's stand-in, not in any restore read in the fine-tuned model.

**Against `GOALS.md`'s question, under the plan's three answers.**
- *Yes, VPD can tell them apart in the real model*: not reached. No real-model number separated the known-kind anchors.
- *Undecidable with VPD*: does not fit as the plan defines it. That answer needs the toys to separate and the real
  updates to sit between; here the toys did not separate.
- *No, the anchors read alike in the real model*: literally what was found. But neither real-model number passed a
  control showing it could separate kinds on any update. The honest reading is that the instruments were unread, not that
  the answer is No. That outcome is outside the plan's tree, which sends it to the user.

**The method's standing limit** (`decisions.md`, "Limit of the elicitation reading"). A new mechanism built from
subcomponents the original model already has, put together under new gating, reads as "elicited" to VPD, because its
unit is the rank-one subcomponent, not the circuit. So "elicits" from VPD means "uses pieces the original model already
has", never "uses a circuit it already runs". The only check that narrows this is a Γ from a second decomposition seed.
Nothing in E50 narrows it.

## How compelling is the evidence

Two seeds and four tables, all one key-to-word task family on the 67M model, scored at one position. The two seeds
agreed within 0.02 on every alive subcomponent and within 0.06 on S, against a separation bar of 0.3. The restore's
rounding is about 0.0015 of the gain. The class gaps sit close to zero, not near the bar, so "does not separate" is not a
threshold case. But neither number has a control that reads low, so the result says the two kinds read alike on these
instruments, not that the instruments could have told them apart. Every alive subcomponent cannot see what the original
weights relay; S misses subcomponents Γ moves by 0.3 or less; both read the key position only. The copy anchor itself is
weak, as above.

## Surprises

1. **The random maps read like the copies on every alive subcomponent, 0.66 to 0.73 against a filed ceiling of 0.2.**
   The filed lean, that every alive subcomponent separates the kinds, was missed. The algebra says what the number
   measures; why both kinds are mostly relayed is not yet known.
2. **S on the copies took back under a fifth**, missing its filed floor of 0.2 on all four; E38's triggers had read about
   a fifth.
3. **S took back the same 2.0 to 2.7 nats on every fine-tune**, whatever its size or the gain. A shared task-format part
   is a candidate.
4. The stand-in ratios, copied from E46, sat just outside their filed ranges on two fine-tunes, 0.356 and 0.414. Narrow;
   one line.

## What next

1. The write-only restore check, $0, two minutes, reported as information: whether the every-alive share depends on the
   decomposition at all.
2. E50 Amendment 1, being drafted now and filed before it runs: a $0 power amendment, the card's recommended option.
3. The re-gating line's direction after E46 Amendment 3 and E50: the decision card below, chosen after Amendment 1 reads.

## Pending decision (card text)

**What should the re-gating line do next, now that no real-model number tells known-existing from known-new fine-tunes,
and neither has been shown able to?**

*Evidence.* On E12's toys, the stand-in ratio separates copies from random maps: about 0.9 against about 0.4. It is read
in Γ's stand-in model, not the fine-tuned model. E46 Amendment 3 found it is nearly the gate gain over the split's gain.
The update's own subcomponents under Γ add only a few hundredths. A hand-made write with no learned machinery reads near
the new kind, 0.40 and 0.57. E50 then read the same toys inside the fine-tuned model. Restoring every alive subcomponent
took back three quarters of the gain on both kinds. By algebra it measures how much of the gain the original weights
relay. Γ's switched-on set took back a tenth to a fifth on both kinds, alike, and its copy floor was missed. Neither
real-model number has a control that reads low, so neither has been shown able to separate kinds. One caveat bears on
every option. The original model scores the copy answers only 3 to 4 nats better than the random ones. So the copy
anchor is weak, and both fine-tunes may mostly learn the shared format.

*Options.*
1. **Carry the stand-in ratio to Qwen alone.** This is the plan's filed branch for "does not separate": stage 1 on
   Qwen3-1.7B, about $5 to $10, a gated run, with every sentence saying the reading is Γ's and has no real-model check
   behind it on the 67M.
2. **Answer "No" for these toys in the real model, and stop spending on the elicitation question.** Keep E38's
   descriptive result, that Γ finds pieces an update draws on without being told the target, and state the method's
   limit. This departs from the approved tree, which sends "does not separate" to option 1.
3. **A $0 power amendment first, then choose.** Give a real-model number a control shown to work. Read every alive
   subcomponent and S on E46's hand write and random update, as a control that should read low. Split S into the part
   shared by all eight fine-tunes and the kind-specific rest, and read each. This is being drafted now as E50
   Amendment 1, filed before it runs. The user chooses between 1 and 2 after it reads.

*Recommendation.* Option 3. It costs nothing and is the only option that learns whether a real-model number could ever
have answered. If a number shows power and still reads the kinds alike, option 2 has evidence behind it; if none shows
power, option 1 goes ahead knowing the real model gave no check.

*Run line.* E50, Slurm 43040, cluster debug, $0; Amendment 1 being drafted, $0.

## Scorecard

Filed in `predictions.json`; read from `restore_anchors.json`. Rules are first-match as filed.

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| reading | the kinds separate on every alive subcomponent | worst-pair gap ≥ 0.3 on both seeds, every alive or S | gaps 0.028, 0.046 every alive; −0.031, −0.061 S | does not separate; filed lean missed |
| C1 | original model reproduces E46 | T0 within 0.02, path within 1e-3 | 0 difference, four tables | pass |
| C2 | fine-tune reproduces E46 | T1 within 0.02 of the gain | 0 difference, eight | pass |
| R0 | empty set changes nothing | ≤ 1e-5 nats | 0, eight | pass |
| R1 | any set restored in the original model changes nothing | ≤ 0.01 nats | 0, eight, every set | pass |
| R2 | whole output reset gives back T0 | ≤ 0.01 nats | 0, eight | pass |
| copy, every alive | ≥ 0.5 | reported | 0.757-0.781 | met, four of four |
| copy, S | ≥ 0.2 | reported | 0.138-0.190 | missed, four of four |
| copy, gold | ≥ 0.1 | reported | 0.101-0.118 | met, four of four |
| copy, D | ≥ S − 0.05 | reported | 0.389-0.474 | met, four of four |
| random map, every alive | ≤ 0.2 | reported | 0.664-0.730 | missed, four of four |
| random map, S | ≤ 0.05 | reported | 0.097-0.204 | missed, four of four |
| random map, gold | ≤ 0.1 | reported | 0.037-0.171 | met on table B, missed on table A |
| random map, D | ≤ 0.2 | reported | 0.133-0.240 | met three of four; table A seed 0 0.240 |
| random sets | mean ≤ 0.05, max ≤ 0.15 | reported | mean ≤ 0.038, max ≤ 0.112 | met, eight of eight |
| stand-in ratio | copy 0.87-0.97, random map 0.36-0.41 | reported, copied | 0.356 (random map A seed 1) and 0.414 (random map B seed 0) | met six of eight, narrow misses |

## Not run

- The write-only restore check; it tells whether the every-alive share depends on the decomposition, not its power.
- Any update known to read low on either real-model number: E46's hand write or random update. This is E50 Amendment 1.
- S split into its shared core and its kind-specific rest.
- A copy anchor the original model already performs in this format.
- Any position other than the key position; the restore at earlier positions or per position.
- A per-member normalisation of S, given its fourfold size difference between kinds.
- A second decomposition seed's Γ, the one check that narrows the method's limit.
- Qwen, any other task family, more than two seeds.

## Provenance

Slurm 43040, MATS cluster L40, debug QoS, 78 s, $0. Log `cluster/logs/commute-restore-anchors-43040.out`; run log in
`experiments/restore_anchors/results/large/run.log`. States: `experiments/regating_table/results/large/`, checked
against `stage0a.json`'s sha256. Alive set: `experiments/transport_test/results/large/alive.pt`. Gold positions in the
first 2,000 Pile validation rows: 10,232 and 8,439 for the copy tables, 1,062 and 1,160 for the random maps.
