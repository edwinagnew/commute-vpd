# What the 67M results let a paper say about teacher training and reuse

*2026-11-07, regating-lead's synthesis writer. No new run and no new figure: every number is quoted from a report or
result file named in a "Sources" line. Built from Fable's draft and advice of 2026-11-06, corrected by the last three
results of that night. Answer paragraph revised after an independent Opus check (2026-11-07; three of five sentences
were overstated: removal-test labels, teacher-training exceptions and "because"; scratch `synth-check/check.md`).
Updated the same day with E130: placement protection varies by copy; and with its Amendment 1: on the step-20 Adam
copy it varies by rows, so its reversal is not established.*

Source keys, used in the Sources lines below: **E121** `reports/analyses/2026-11-06_analysis_e121-copy-gap-ladder.md`;
**E122** `..._e122-damage-crossed-pair.md`; **E123** `..._e123-case-label-both-readings.md`; **E125**
`..._e125-fresh-reuse-pair.md`; **E127** `..._e127-ls-start-faithfulness.md`; **E116** `..._e116-definition-table.md`;
**E119** `..._e119-teacher-route-split-triggers.md`; **E126** `..._e126-copy-reproducibility.md`, all in
`reports/analyses/` dated 2026-11-06; **TT** `reports/analyses/2026-10-23_analysis_teacher-training.md`; **A10**
`reports/analyses/2026-11-02_analysis_e34-amend10-input-weighted.md`; **E130**
`reports/analyses/2026-11-07_analysis_e130-copy-protection.md`; **claims** `docs/research_log/parts/sections/claims.html`.

We gathered every 67M result on carrying a fine-tune into the released decomposition, and on reuse read two
independent ways. Teacher training carries most narrow fine-tunes faithfully, with named exceptions, and fails on
one broad fine-tune whose copied change hides most but not enough of its excess. How much teacher training's placement
hides is not a constant of the recipe: it is steady on two lowercase copies, but on one it ranges from partial to
none depending on the rows read. For `GOALS.md` this gives a paper a
carrier for narrow updates with known limits, and no 67M case where the two readings of reuse agree on a positive.

## 1. The answer

At 67M no fine-tune has yet been labelled "reuses C" under the frozen definition: the two label pairs it has read both
read "not through C" on every state. On those pairs the route split and its teacher-trained copies give each real
fine-tune the same masked-off label as its memorising twin, so they agree with the definition only on a negative that
cannot tell the arms apart, while the decomposition's removal test reads "through" on most states of both arms. On
the narrow trigger tasks, teacher training copies the behaviour with Δ frozen and passes the check, with a rare-target
task and some Pile-text copies as exceptions; where it was read (the cipher, two label pairs), the copied behaviour sits
on subcomponents Γ keeps off. On the broad
lowercase full-matrix Adam fine-tune its copy fails (1.91 against a fail line of 1.61); the copy's own weight change
fails worse when moved into Δ (3.53), so its placement removes most of the excess (about 64%), but not enough. On
another lowercase copy the placement's effect ranges from partial protection to none depending on the rows, so the
protection is not a constant of the recipe and where it fails is not settled. One seed, one task; whether fragility is the cause is not tested.

## 2. Teacher training: where it is strong, where it is weak

**Terms.** Each matrix is $W = UV^\top + \Delta$. *Teacher training* trains only $U, V$ to match the fine-tuned model's
outputs, with Γ and Δ frozen. *The check* is VPD's mask adversary on Pile text, as a ratio to W₀'s own reading: 1 means
as hard to break as the released decomposition. *Do-nothing* leaves the update δW in Δ.

**Strong on narrow fine-tunes.** On six trigger fine-tunes every copy keeps a held-out target probability of 0.90 to
1.00. Under an adversary eight times longer, the twelve copies read 0.95 to 1.15. Left in Δ, five of the six fine-tunes
read 2.4 to 10.4 there, so on these tasks the check has teeth. On the narrow emoticon task under the usual 20-step
check it has few: do-nothing itself reads 1.39, near the copies' 0.97 to 1.00.

Sources: claims:58; TT:399-400, TT:1124-1128; E121:205-212.

**Exceptions on narrow tasks.** The rare target task reads "neither": its task-text copy keeps a target probability of
0.808, just under the kept line of 0.813. Its Pile-text copy fails the check at 2.48. A rank-8 LoRA copied from Pile
text learns only part of the behaviour, with target probabilities of 0.52 to 0.88. The emoticon Adam Pile-text copy is
unstable across batch orders and reaches only 0.34 to 0.38 of its teacher's gain. A copy stopped at step 10 reads
inconclusive at 1.547. Plain SGD at 0.03 as the copy's optimiser fails on the Adam and LoRA teachers, at one rate and
step count.

Sources: `experiments/TRIED.md:64`, `:87`, `:96`; E126:16-19.

**Where the copied behaviour sits.** "Into the decomposition" holds at full masks. Where it was read, the copies carry
the behaviour on subcomponents Γ keeps off. Under Γ's own masks at the trigger, the cipher copies lose it: target
probability 0.007 and 0.001 against 0.964 and 1.000 unmasked. All eight label-pair copies read masked-off.

Sources: `experiments/TRIED.md:86`; E123:88-93; E125:201-213.

**Weak on the broad lowercase fine-tune.** The copy of the full-matrix Adam fine-tune fails at 1.91 against a fail line
of 1.61. The copy of a rank-8 LoRA fine-tune of the same task passes at 1.33. Crossing the two fine-tunes' damage to
plain text with their recipe did not separate the causes: both crossed copies sat within a floor of a line.

Sources: E122:91-94, E122:197-200; E127:312-314.

**The first gap: why it fails, as the evidence now stands.** The figure
`experiments/ls_start_faith/figures/amend2_delta.png` reads each copy's own weight change twice, at the same weights:
riding the alive subcomponents, and moved into Δ.

| lowercase state | on the subcomponents | the same change in Δ |
|---|---|---|
| do-nothing, the fine-tune's own change | | 4.79 |
| least-squares start | 8.33 | 2.95 |
| copy trained from that start | 9.40 | 3.96 |
| standard teacher-trained copy | 1.91 | 3.53 |
| copy of the rank-8 LoRA | 1.33 | 2.34 |
| copy of the full Adam fine-tune stopped at step 20 | 2.84 | 2.17 |
| copy of a rank-8 LoRA trained to step 2,400 | 1.53 | 3.17 |

Sources: E127:361-367; `experiments/ls_start_faith/results/ls_start_faith_amend2.json` → `score.ratio`; E130:87-92, E130:154-162;
`experiments/copy_protection/results/copy_protection.json` → `score.ratio`;
`experiments/copy_protection/results/copy_protection_amend1.json` → `score.overall`, `score["block rows *"]`.

**Reading it.** Every copy's change fails once it sits in Δ. Placement cuts both ways. The least-squares split's
placement multiplies the excess about four times. Teacher training's placement divides it on three of the four
lowercase copies read, by 2.8 for the standard Adam copy and about 4 for the two LoRA copies. On the copy of the Adam
fine-tune stopped at step 20 it multiplied the excess by 1.6 on rows 0-255: 2.17 in Δ became 2.84 placed. On a second
set, rows 256-511, the same copy's placement cut the excess partly, to 0.78 of the in-Δ excess. Per block of 128 rows
it reads 0.62, 1.75, 0.60 and 0.97, so only one block adds excess. The LoRA copy trained to step 2,400 reads 0.23 to
0.28 on all four blocks. So teacher training's placement divides the excess on both LoRA copies, stably across row
sets, and on the standard Adam copy, read on rows 0-255 only. On the step-20 Adam copy it ranges from partial to
none depending on the rows. The protection is not a constant of the recipe, and where it fails is not settled. One
seed each; damage to plain text does not order the copies. In nats it takes the Adam copy's excess from 1.75 to 0.63, removing about 64%: most of it,
but not enough to pass. The LoRA copy's change also fails in Δ, at 2.34, yet passes once placed. So fragility in Δ does
not separate the failing copy from the passing one; what is left after placement does. "Fails because the function is
fragile" is therefore an accounting split of one copy's excess, not a tested cause. We predicted the Adam copy's change would pass in Δ at 1.24; it
read 3.53, so this reverses the filed lean. Teacher training's loss is the KL from the fine-tuned model alone, with no
mask, adversary or faithfulness term. Its placement is therefore an accident of training at full masks, protective on
three copies of four and row-dependent on the fourth, not by design. Why it protects when it does is a hypothesis, untested: its gradients may fall on subcomponents Γ keeps on, which the adversary cannot
switch off. That sits uneasily beside the label-pair copies, which carry their trigger gain on subcomponents Γ keeps
off; the two are read at different positions and tasks. One caveat bounds "fragile": in Δ the change shares one mask
per matrix, so fragility is read under that placement only. One seed, one task, one card.

Sources: E127:355-359, E127:371-377, E127:379-386, E127:388-398, E127:422-428; `src/commute/teacher.py:40-47`;
`src/commute/row_copy.py:81-82`.

## 3. The second gap: why a copy sits far from the fine-tune's weight change

On emoticon the copy's weight change sits at an input-weighted cosine of 0.40 to the fine-tune's, where a rerun sits at
0.83. The gap is not emoticon's alone: on two map tables the copies read 0.57 and 0.58 against reruns' 0.75 and 0.76.

Sources: A10:35-39; E121:27-29.

**What is shown.** The figure `experiments/copy_gap_ladder/figures/ladder.png` changes one difference at a time.
Training only $U, V$ costs 0.28 of cosine and the lower learning rate 0.19. Learning the fine-tuned model's outputs in
place of the labels costs nothing. The copy is a different change, not the fine-tune's plus an idle part: without its
remainder, the target probability falls to about 0.29. It also moves the trigger's logits less like the fine-tune,
0.64 against a rerun's 0.86. A close copy exists inside $U, V$: started from the least-squares split, the same
training ends at 0.81 and keeps the behaviour.

Sources: E121:76-85, E121:136-140, E121:163-165, E121:288.

**What is hypothesis.** The close copy is less faithful than the far one. On emoticon it reads 1.55 to 1.57, against
1.39 for do-nothing and about 1.0 for standard copies, a gap under the floor of 0.21. On lowercase the copy from that
start fails at 9.40. The reading that fits is that the copy's change is shaped to sit where the masks protect it. It is
a hypothesis: the cosines are on emoticon, the placement result is on lowercase, and emoticon's copy has about 0.005
nats of excess, too little to read. A rival is open too: the lower-rate students end at half δW's size, and longer runs
were not tried. The earlier reading, that the copy drops δW's collateral change on ordinary text, fails on lowercase:
the copy's change in Δ keeps 67% of do-nothing's excess.

Sources: E121:87-88, E121:204-214; E122:96; E127:330, E127:404-409.

## 4. Reuse at 67M, read both ways

**The two readings.** The *definition* is `decisions.md` rows 81 and 87, frozen by row 164. A Pile-trained linear
readout must hold the capability. The fine-tuned map must be a small function of it that carries to held-out values.
Removing the readout's direction must break the behaviour while sparing a twin. The *decomposition's* readings are
the route split and the removal test. The route split's re-switching share $r$ is the part of the gain carried when Γ
switches on released subcomponents. Its masked-off share is the part carried by subcomponents Γ keeps off.

| fine-tune pair | definition | route split, fine-tunes and copies | removal test |
|---|---|---|---|
| case label, real | not through C | masked-off | seeds disagree: unclear, through |
| case label, memorising twin | not through C | masked-off | through |
| repeat label, real | not through C | masked-off | through |
| repeat label, memorising twin | not through C | masked-off | through |
| copy maps against random maps | unreadable at possession | re-switching against masked-off | |
| ten older tasks, pronoun | unreadable or does not apply | | |

Sources: E123:88-93, E123:136-141, E123:152-157; E125:123-128, E125:201-213, E125:233-238; E116:17-22; `experiments/TRIED.md:34`,
`:239`; `decisions.md:81`, `:87`, `:164`.

The figure `experiments/fresh_reuse_pair/figures/route.png` shows all four repeat-label fine-tunes in the masked-off
corner, beside the case-label pair.

**Reading it.** The route split agrees with the definition only on the real arms of the two label pairs: neither finds
them running through existing structure. That agreement has no teeth, because both give the memorising twin the same
label. The removal test disagrees on the positive side. It reads "through", a reuse-like label, on seven of eight
states, the twins highest, where the definition reads "not through C" on all eight. The eighth, the case-label real
arm's first seed, reads "unclear", so that arm's seeds disagree and the filed joint with the definition is "reversed". The decomposition's one
positive sorting, copy maps against random maps, sits on tasks the definition cannot read. The definition's removal
clause fails by a wide margin: the repeat-label fine-tune loses 0.021 to 0.045 of its gain at five sites, against a bar
of 0.50. The definition's side does separate real from twin, through carrying and a removal share four times the
twin's, but under its own bar. So no 67M state reads "reuses C" under the frozen definition, across eight label states.

Sources: E125:113-117, E125:152-158, E125:183-189, E125:274-277; E123:166-173.

**Two limits on any positive the decomposition gives.** First, $r$ is set by the task and trigger, not the update: a
fine-tune that made held-out text worse read 0.82. Second, the elicitation-granularity limit: VPD's unit is the
rank-one subcomponent, so a new circuit built from old subcomponents under new switching reads as elicited.

Sources: claims:22; `decisions.md:73`; E125:299-302.

## 5. The strongest honest case, and its red team

**The case.** On most narrow trigger tasks, teacher training copies the behaviour into the subcomponents and keeps the
check at W₀'s level under a long adversary where do-nothing fails. Its exceptions are named: a rare target, some
Pile-text copies, a step-10 copy, plain SGD. Its known failure is a broad full-matrix fine-tune, where its placement
removes about 64% of the excess, not enough to pass; on another lowercase copy its placement ranges from partial
protection to none depending on the rows. The frozen Γ sorts maps built to reuse from maps built to need new ones, and task-text copies keep
their fine-tune's $r$ within 0.06. On the label pairs, the definition and the route split agree that the real
fine-tunes do not run through existing structure.

Sources: TT:1124-1128; `experiments/TRIED.md:64`, `:87`, `:96`; E127:355-359, E127:371-372; `experiments/TRIED.md:34`; E119:74-80; E123:166-170.

**Red team.**

- *The check is one 20-step local adversary, with lines we set.* Holding the alive masks at 1 is a setting the free
  search may choose, yet held it reads W₀ at 1.60 nats against 0.69 free. Open: a stronger free adversary is not run.
- *Faithfulness by hiding.* Teacher training passes by where it puts the change, not by carrying a robust function. Open:
  whether the placement sits on subcomponents Γ keeps on is not read, and no loss term asks for it; on one lowercase
  copy of four the placement ranges from partial protection to none depending on the rows (E130 and its Amendment 1).
  Where the trigger
  was read, the copies carry their gain on subcomponents Γ keeps off.
- *"Fails because fragile" is an accounting split.* Conceded: the LoRA copy's change also fails in Δ yet passes placed.
- *The readings disagree on positives.* Conceded: the removal test reads "through" where the definition reads "not
  through C", on seven of eight label states.
- *Pass without teeth on narrow tasks.* Partly answered: under the long adversary do-nothing fails on five of six.
- *The copy is a different update.* Partly answered: task-text copies keep their fine-tune's $r$, on one seed.
  Pile-text copies sit higher, and one is unstable across batch orders, with a spread of 0.135.
- *$r$ reads the task, not the update.* Answered against the claim: $r$ describes what Γ can route at a trigger.
- *No reuse is detected.* Open. Every route-split positive is by construction and unreadable to the definition. Every
  definition-readable pair is a negative where the route split reads real and twin alike.
- *The twin's update is larger.* Partly answered. It is 1.6 times the real's on the repeat pair and 3 times on the
  case pair, but the removal share does not rise with size across the eight states.

Sources: E127:135-141, E127:224-228, E127:357-359, E127:535-536; E125:229-231, E125:275; `experiments/TRIED.md:86`; TT:399-400; E119:74-80; E126:16-19; claims:22; E125:243-251.

## 6. Open gaps that decide the paper, each with the run that would close it

- **The first gap, the broad failure.** Teacher training with VPD's masked reconstruction added to its loss, on the lowercase
  Adam fine-tune. A pass makes placement the lever for broad updates and turns the accounting split into a tested
  cause; a fail makes the function the limit.
- **The second gap and the hiding objection.** Read where the standard copy's change and the least-squares start's sit
  against Γ's causal importances on the check's rows. Read it on both tasks. High-importance placement for the copy alone confirms the
  hypothesis.
- **The check's strength.** A free adversary with more steps or restarts at W₀. If it reaches the held 1.60 nats, every
  ratio so far is on an under-reading scale.
- **Reuse detection.** No run under the frozen definition is in hand. The repeat-label fine-tune's flat removal share
  across sites leaves two readings, little use or use at many sites. A removal at all sites at once would tell them
  apart, but adding it to a verdict is a definition change, which only the user makes.
- **The masked-off corner.** A label pair whose labels W₀ already predicts somewhat. If it leaves the corner, the corner
  is about starting log-odds, not label tasks.

Sources: E127:463-472, E127:468-470; E125:175-181, E125:324-331.

## Departures from the literature

- **The check.** The VPD paper reports the adversarial loss, not pass lines; the lines, the floor and one adversary per
  128-row block are ours. Source: E127:535-536.
- **Reading a change in Δ** at the same weights is ours, to compare placements. Source: E127:540-542.
- **Teacher training is not VPD's component training**, which trains with faithfulness, stochastic and minimality
  losses and Γ alongside. Its KL loss also departs from Hinton et al. 2015: temperature 1, no hard-label term. Source:
  E121:308-314.
- **Jain et al. 2024.** Definition 1 uses a binary ridge readout with a per-type control task. Definition 2 adds row 87's
  held-out carrying and the direction-removal check. Source: E125:384-387.
- **The route split and removal test** are this project's measures, not VPD's. Source: E125:391-392.

## Not run

- Teacher training with any mask or faithfulness term in its loss, at 67M.
- Where any change sits against Γ's causal importances.
- A stronger free adversary at W₀; adversary seeds 1 and 2 for the Δ placement reads; Pile set B for them.
- A weight cosine between the standard or E127's LoRA lowercase copy and its fine-tune; E130's two copies read 0.58
  and 0.59, input-weighted.
- Longer lower-rate students, or teacher training at the fine-tune's rate.
- A removal at several sites at once; a size-matched twin; a label pair with higher starting log-odds.
- The rank-8 LoRA's own do-nothing on the L40.
- An independent check of this synthesis.
