# E123 plan: the case-label pair read by both readings

*Written 2026-11-06 from Fable's draft `~/commute-data/scratch/regating-lead/e123-fable/plan_draft.md`, adopted by the
lead for the case-label pair, trimmed, with the fresh-pair option moved out (it is E125). Nothing run. Apparatus,
timing sources and the list of changes from the draft: `2026-11-06_plan_e123-case-label-both-readings_appendix.md`.
Filed rules: `experiments/case_label_both/results/predictions.json`.*

**In plain words.** One small-model task has been read by the independent reuse definition: the case-label pair, where
a fine-tune learns to answer one of two words depending on whether the previous word is capitalised, beside a twin that
can only memorise. The decomposition has never read that pair. This plan reads E112's four saved fine-tunes through the
decomposition's three filed readings, so the paper can say for the first time what both readings say about one task.

## Request in the Terms' words

On E112's four saved fine-tunes of the 67M model (real s0/s1, twin s0/s1; no fine-tune rerun): (1) the route split of
each fine-tune, do-nothing and least-squares split, labelled by E12's filed lines; (2) component training with loss
KL(fine-tuned model ‖ student), Γ and Δ frozen, E34's arm-A recipe (teacher training), each student read by the route
split on its own state, same lines; (3) E38 Amendment 2's removal test of the recruited set S at the trigger. Reported
beside the definition's verdicts, which stay E112's. No rule is added to either side.

## Design check (`/design`)

- **Goal.** `GOALS.md`, the question made precise: whether the decomposition's account of an update agrees with an
  independent account. The paper has no task where both stand (Fable's strategy advice §1: "The two readings have
  never both stood on one task").
- **Nearest tried.** E112 and E113 (TRIED.md rows E112, E113) gave the definition's side: both arms "not through C".
  E114 read the weight-side locator on the same states. E119 read teacher-trained students on three trigger tasks.
  None put a decomposition reading on this pair (`experiments/SAVED.md`, case_label section: only E112's files).
- **Patch or step.** A step: it fills the one empty cell that the paper's re-use sentence needs.
- **Decisions relied on.** Rows 81, 87, 164, 165 (definition frozen); the elicitation-granularity limit. No departure.
- **Under an adopted definition, unchanged?** Yes. The definition's verdicts are E112's, quoted, not re-read. Row 81:
  "Def 2: fine-tuning reuses C if the post-trained map is g∘f_C … Added: reuse needs a causal claim (ablating or
  patching the readout direction breaks the fine-tuned behaviour …)". Row 87: "a fine-tune reuses capability C only if
  g, learnt on the fine-tuning examples, also works on held-out values". The decomposition side uses E12's, E119's and
  E38 A2's filed lines unchanged; the joint table below names outcomes, it is not a verdict.
- **Class.** Inside the user-approved task ("Build one task both read", 2026-11-06): saved states, existing harness,
  $0. Goes ahead without further sign-off.
- **What would change what we do next.** The joint outcome picks the paper's sentence (table below). "Separates as
  built" or "reversed" would be news either way; "no label" is the modal outcome and closes the 67M side of the
  question for this pair.
- **Control first.** Stage 0, alone, stops the job on any miss: the moved rules on saved files, W₀'s accuracy and
  every state's gain against E112's and E114's numbers, the two forward paths at W₀, and E38 A2's removal controls.

## The pair

E112's real arm says label A after a capitalised word, B otherwise; the twin gives each word form A or B by a seeded
shuffle, so only memory can learn it. Same trigger " \+", same labels " carpet" / " camel". Known confound, carried
into every sentence: the twin trained 10× longer (step 300 against 30) and its update is about 3× the real's (summed
Frobenius 13.3 / 12.6 against 4.6 / 4.7, E112's `stage1.json`).

All decomposition reads use the first 256 of E112's 1,024 fit windows: train-pair words in rows the fine-tune never
saw. Not the held-out-pair windows: the twin is at chance there (0.506 / 0.516), so its gain there is toward either
label. E114 made the same choice. The window's own label is a 42nd column that neither the model nor Γ ever sees.

## Stages and expected values

One MATS debug L40 job, about 17 min, exit 3 for a resubmission if a student would end past 25 min.

| stage | what | expected | stop |
|---|---|---|---|
| 0 | moved route rule on E12 A1's students; E38 A2's seed classes reread | 4 copy re-switching, 4 maps masked-off; classes equal the file | any miss |
| 0 | W₀ held-out accuracy | 0.5206 real, 0.4969 twin (E112) | off by > 0.000625 |
| 0 | fit-window gains per state, label gap and log-odds | 12.83 / 12.04 / 10.23 / 8.80 and 23.29 / 24.52 / 23.63 / 22.45 nats (E112, E114) | off by > 0.02 |
| 0 | route path against removal path at W₀ | equal (E38 read 0) | > 1e-3 |
| 0 | E38 A2's C1-C4 per state | C1-C3 read 0; C4 KL ≥ 1 where the busy set has ≥ 100 pieces | any miss |
| 1 | route split of each fine-tune | do-nothing e_gate = −r; least-squares r = do-nothing r (1e-6); r 0.45-0.65; label neither | the two identities |
| 2 | removal of S at the trigger | S nonempty; random D within ±0.16; class not | none |
| 3 | teacher-trained student per state | fit accuracy ≥ 0.90; gain ratio 0.9-1.05; r within 0.06 of the fine-tune's; label neither | none |
| 4 | per arm: the two seeds' common class; joint per measure | route "no label", copy "no label", removal "real new-like" | none |

At W₀ the route shares are not read (gain 0). T₀ on the read windows is not on file, roughly −18 to −21. For the
do-nothing reference the identities above are exact by construction. The route rule's positive control is E12
Amendment 1's students, built to separate; the removal's is C4 and E38 A2's own records; the copy's is the fine-tune's
own fit behaviour on the same path.

## What each joint outcome means

The definition reads both arms "not through C". Per decomposition measure, from the two arms' sides (reuse-like:
re-switching or through; new-like: masked-off or not):

| joint | when | sentence for the paper |
|---|---|---|
| separates as built | real reuse-like, twin not | On the one 67M pair both readings can see, the decomposition sorts the fine-tunes as they were built, while the definition finds the case feature in neither one's causal path. The twin's update is 3× larger, so the split may be size. |
| real new-like | real new-like, twin not reuse-like | Both readings say the case-label fine-tune does not run through existing structure the way the decomposition reads reuse; neither tells it from a memorising twin. |
| no label | neither reuse-like nor new-like on the real arm, twin not reuse-like | At 67M the decomposition gives no reuse label on the only pair the definition reads. |
| both reuse-like | both arms reuse-like | The decomposition reads reuse on a twin built to memorise: it has no teeth on this pair. |
| reversed | twin reuse-like, real not | The decomposition's label follows the update's size or the trigger, not its kind; the paper should not call it a reuse reading. |
| unreadable | a side cannot be read | Reported; the definition's reading stands alone, as now. |

**Blind spots of the comparison.** The definition names a capability C, the case readout; the decomposition names
none. So even "real new-like" is two labels about different objects lining up, not one claim confirmed twice. A new
circuit built from old pieces reads "elicited" to VPD.

## Noise

Route split: E12 Amendment 1's batch-order noise (|Δ e_gate| 0.002, |Δ L01| / gain 0.0006); the two seeds per arm
give a seed spread; the 256-window gain against E114's 1,024-window gain gives the window spread. Removal: the 20
random sets per state are the null. Copy: E119's arm-A students reproduced within 0.016.

## Departures from the literature

- **Jain et al. 2024 Definition 2.** Row 87's held-out carrying, as E112 filed it; nothing further.
- **VPD.** The route split, teacher training and the removal test are this project's measures, not the VPD paper's.
  Γ sees 41-token windows where it trained on rows of 512.
- **E34 and E119.** No teacher regeneration (the teacher is E112's saved change); arm A only; E112's windows with a
  per-window label column; 256 read windows instead of 73-103.
- **E38 Amendment 2.** No null set (no different-target fine-tune at this trigger), no busy-set D, no all-position
  removal, no census; fit windows instead of held-out ones.

## Not run

- Arm B (Pile-text students); a second student seed or batch order.
- E38 A2's null set, busy-set D, all-position removal and census.
- The route split on held-out-pair windows.
- Any fresh fine-tune or second pair (E125); any change to the definition side.
