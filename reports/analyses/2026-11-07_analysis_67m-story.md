# What the 67M results let a paper say about teacher training and reuse

*2026-11-07, regating-lead's synthesis writer. No new run and no new figure: every number is quoted from a report or
result file named in a "Sources" line. Built from Fable's draft and advice of 2026-11-06, corrected by the last three
results of that night. Provisional: no independent check.*

Source keys, used in the Sources lines below: **E121** `reports/analyses/2026-11-06_analysis_e121-copy-gap-ladder.md`;
**E122** `..._e122-damage-crossed-pair.md`; **E123** `..._e123-case-label-both-readings.md`; **E125**
`..._e125-fresh-reuse-pair.md`; **E127** `..._e127-ls-start-faithfulness.md`; **E116** `..._e116-definition-table.md`;
**E119** `..._e119-teacher-route-split-triggers.md`; **E126** `..._e126-copy-reproducibility.md`, all in
`reports/analyses/` dated 2026-11-06; **TT** `reports/analyses/2026-10-23_analysis_teacher-training.md`; **A10**
`reports/analyses/2026-11-02_analysis_e34-amend10-input-weighted.md`; **claims** `docs/research_log/parts/sections/claims.html`.

We gathered every 67M result on carrying a fine-tune into the released decomposition, and on reuse read two
independent ways. Teacher training carries narrow fine-tunes faithfully, fails on one broad fine-tune because the
copied function is itself fragile, and lands far from the fine-tune's weight change for reasons partly shown. For
`GOALS.md` this gives a paper a faithful carrier for narrow updates, but no case at 67M where reuse is detected.

## 1. The answer

At 67M we do not yet detect reuse. On the only tasks the frozen definition can read, it finds no reuse. There the
decomposition gives each real fine-tune the same label as its memorising twin, so the two readings agree only on
negatives that cannot tell the arms apart. Teacher training is the strong half: it carries narrow fine-tunes into the
decomposition and keeps it faithful. It fails on a broad full-matrix fine-tune because the function it copies is fragile
under the check, and its placement on the subcomponents hides only part of that.

## 2. Teacher training: where it is strong, where it is weak

**Terms.** Each matrix is $W = UV^\top + \Delta$. *Teacher training* trains only $U, V$ to match the fine-tuned model's
outputs, with Γ and Δ frozen. *The check* is VPD's mask adversary on Pile text, as a ratio to W₀'s own reading: 1 means
as hard to break as the released decomposition. *Do-nothing* leaves the update δW in Δ.

**Strong on narrow fine-tunes.** On six trigger fine-tunes every copy keeps a held-out target probability of 0.90 to
1.00. Under an adversary eight times longer, the twelve copies read 0.95 to 1.15. Left in Δ, five of the six fine-tunes
read 2.4 to 10.4 there, so on these tasks the check has teeth. On the narrow emoticon task under the usual 20-step
check it has few: do-nothing itself reads 1.39, near the copies' 0.97 to 1.00.

Sources: claims:58; TT:399-400, TT:1124-1128; E121:205-212.

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

Sources: E127:361-367; `experiments/ls_start_faith/results/ls_start_faith_amend2.json` → `score.ratio`.

**Reading it.** Every copy's change fails once it sits in Δ, so the broad lowercase function is fragile under the check
whoever carries it. Placement cuts both ways. The least-squares split's placement multiplies the excess about four
times. Teacher training's placement divides it, by 2.8 for the Adam copy and 4.1 for the LoRA copy. That is enough to
pull the LoRA copy under the line, not the Adam copy. We predicted the Adam copy's change would pass in Δ at 1.24; it
read 3.53, so this reverses the filed lean. Teacher training's loss is the KL from the fine-tuned model alone, with no
mask, adversary or faithfulness term. Its protective placement is therefore an accident of training at full masks. Why it
protects is a hypothesis, untested: its gradients may fall on subcomponents Γ keeps on, which the adversary cannot
switch off. One caveat bounds "fragile": in Δ the change shares one mask per matrix, so fragility is read under that
placement only.

Sources: E127:355-359, E127:374-377, E127:379-386, E127:388-398, E127:422-428; `src/commute/teacher.py:40-47`;
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
| case label, real | not through C | masked-off | seeds disagree |
| case label, memorising twin | not through C | masked-off | through |
| repeat label, real | not through C | masked-off | through |
| repeat label, memorising twin | not through C | masked-off | through |
| copy maps against random maps | unreadable at possession | re-switching against masked-off | |
| ten older tasks, pronoun | unreadable or does not apply | | |

Sources: E123:88-93, E123:152-157; E125:123-128, E125:201-213, E125:233-238; E116:17-22; `experiments/TRIED.md:34`,
`:239`; `decisions.md:81`, `:87`, `:164`.

The figure `experiments/fresh_reuse_pair/figures/route.png` shows all four repeat-label fine-tunes in the masked-off
corner, beside the case-label pair.

**Reading it.** The two readings agree only on the real arms of the two label pairs: neither finds them running through
existing structure. That agreement has no teeth, because both give the memorising twin the same label. The removal test
disagrees with the definition, reading "through" on seven of eight states, the twins highest. The decomposition's one
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

**The case.** Teacher training carries narrow fine-tunes into the released decomposition, keeping the behaviour and
the check at W₀'s level under a long adversary where do-nothing fails. Its known limit is a broad full-matrix fine-tune,
and the cause is partly located: the copied function is fragile, and teacher training's placement shrinks its exposure three
to four times. The frozen Γ sorts maps built to reuse from maps built to need new ones, and task-text copies keep
their fine-tune's $r$ within 0.06. On the label pairs, the definition and the route split agree that the real
fine-tunes do not run through existing structure.

Sources: TT:1124-1128; E127:355-359; `experiments/TRIED.md:34`; E119:74-80; E123:166-170.

**Red team.**

- *The check is one 20-step local adversary, with lines we set.* Holding the alive masks at 1 is a setting the free
  search may choose, yet held it reads W₀ at 1.60 nats against 0.69 free. Open: a stronger free adversary is not run.
- *Faithfulness by hiding.* Teacher training passes by where it puts the change, not by carrying a robust function. Open:
  whether the placement sits on subcomponents Γ keeps on is not read, and no loss term asks for it.
- *Pass without teeth on narrow tasks.* Partly answered: under the long adversary do-nothing fails on five of six.
- *The copy is a different update.* Partly answered: task-text copies keep their fine-tune's $r$, on one seed.
  Pile-text copies sit higher, and one is unstable across batch orders, with a spread of 0.135.
- *$r$ reads the task, not the update.* Answered against the claim: $r$ describes what Γ can route at a trigger.
- *No reuse is detected.* Open. Every decomposition positive is by construction and unreadable to the definition, and
  every definition-readable pair is a negative where real and twin read alike.
- *The twin's update is larger.* Partly answered. It is 1.6 times the real's on the repeat pair and 3 times on the
  case pair, but the removal share does not rise with size across the eight states.

Sources: E127:135-141, E127:224-228, E127:535-536; TT:399-400; E119:74-80; E126:16-19; claims:22; E125:243-251.

## 6. Open gaps that decide the paper, each with the run that would close it

- **The first gap, the broad failure.** Teacher training with VPD's masked reconstruction added to its loss, on the lowercase
  Adam fine-tune. A pass makes placement the lever for broad updates; a fail makes the function the limit.
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
- A weight cosine between any lowercase copy and its fine-tune.
- Longer lower-rate students, or teacher training at the fine-tune's rate.
- A removal at several sites at once; a size-matched twin; a label pair with higher starting log-odds.
- The rank-8 LoRA's own do-nothing on the L40.
- An independent check of this synthesis.
