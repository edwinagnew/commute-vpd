# E42: independent check of the teacher-training-on-a-new-behaviour analysis (review)

*2026-10-28, independent agent, read-only; no model loaded, no Python run. Numbers re-read with `jq` from
`experiments/teacher_training/results/teacher_training_e42_{rand32,ident32,rand1}.json` and with `grep` from
`cluster/logs/commute-teacher-e42-{40166,40194,40195}.out` (the trigger P cells exist only there). Code read:
`experiments/teacher_training/newskill.py`, `src/commute/kv_windows.py`, `local_faith.route_split` / `route_cells` /
`masked_lo`, `faithfulness.faithfulness`, `harness.pile_ratios`, `faith_score.state_verdict` / `w0_check`,
`teacher.teacher_loss` / `heldout_gap`, `run.record` / `train_student`. Against `GOALS.md`, the plan,
`predictions_e42.json` and `reports/decisions/2026-10-19_decision_success-criterion.md`. Checked: the analysis
`reports/analyses/2026-10-28_analysis_e42-teacher-new-behaviour.md` and the markdown of
`notebooks/E42_teacher_new_behaviour.ipynb`.*

Every number in the analysis's tables matches the results files and logs, and every prediction is scored by its
filed rule. The run's own `score` blocks agree. The headline pass is real but should name the checks it rests on, since
the sparsity loss was not read. Three claims go further than the evidence. The first is that the trigger gap "belongs to
the task with Γ frozen". The second is that the Pile-text copy "picked up the leak". The third is the draft card's
picture of two equal texts in `GOALS.md` that disagree. The text itself already says which of the two decides.

## 1. Verdict per claim

| # | Claim (analysis wording) | Verdict | Why, and the wording I would allow |
|---|---|---|---|
| 1 | "Yes by the paper's checks": the task-text copy keeps the behaviour, passes and leaves Δ untouched, on all three tasks | **Supported, slightly overstated** | P 0.964 / 1.000 / 0.983; adversarial ratios 0.960 / 0.989 / 0.957 and stochastic ratios 0.936 / 0.941 / 0.929, all under both cards' pass lines; Δ frozen (share ≤ 1e-9). But `GOALS.md` → "Faithful" lists five checks, and only four were read. VPD's sparsity loss was not read. E10 and E11 found that with Γ frozen it depends on W′ only, which stops it telling apart maps with the same W′. The copy's W′ is not the fine-tune's, so for the copy it is still a reading. The adversary ran one 20-step start on 256 rows. Allow: "passes the four reconstruction checks the harness runs on 256 Pile validation rows (stochastic, adversarial with one 20-step start, CI-masked, unmasked), with Δ frozen; VPD's sparsity loss was not read." Drop "every check the paper runs" from the card. |
| 2 | "the CI-masked model loses the behaviour at the trigger" (random ciphers) | **Supported** | Log cells: rand32 copy 0.964 full, 0.952 unmasked, **0.007** CI-masked; rand1 1.000 / 1.000 / **0.001**; ident32 keeps 0.929. Δ is out of both the trigger CI-masked model and the harness's Pile CI-masked model (`make_mask_infos(ci.g)` with no weight deltas), so the two readings are the same model on different inputs. In log-odds the masked model still keeps 14% (rand32) and 5% (rand1) of the gain (d + r + i), so "almost all" is right in probability, not "all". Allow as written, adding "mean over 97 held-out windows, one batch order". |
| 3 | "So with Γ frozen this belongs to the task … not specific to teacher training"; "a property of the task under a frozen Γ, not of the map" | **Overstated** | The maps' CI-masked readings are all low (0.016 / 0.017 / 0.007 on rand32), but not for one shared reason. For do-nothing the gain is in Δ: share 1, unmasked P 0.000. For the rand32 least-squares split it is mostly in Δ: share 0.52, unmasked P 0.22. Only the copy and the rand1 least-squares split are "masked off", which matches E10's label-trained component training. So the evidence shows the signature is not specific to teacher training. It does not show that a frozen Γ forces it. No map with Γ frozen has been trained against the masked losses on the task's own windows, e.g. `masked_teacher.py` (E37 Amendment 1) on the cipher windows. No map limited to the subcomponents Γ switches on at the trigger has been tried either. Allow: "With Γ frozen, no map read so far keeps the cipher in the CI-masked model at the trigger: teacher training and label-trained component training carry it through subcomponents Γ keeps off there, and do-nothing leaves it in Δ. So the gap is not specific to teacher training; whether any frozen-Γ map can close it is untested." |
| 4 | "Pile text cannot show it: the marker occurs at none of the positions in the first 1,024 Pile-text rows" | **Wrong rows** | `pile_reading` scanned the first 1,024 of arm B's *training* rows (from 32,768). The harness reads 256 *validation* rows, which were not scanned. The conclusion is very likely still true, because the marker was chosen for zero count in the source rows, but the cited evidence is for other rows. Allow: "The marker did not occur in the 1,024 Pile training rows scanned and was chosen for having zero count in the windows' source rows; the harness's 256 validation rows were not scanned." The same fix applies to "where the marker occurs at none of the Pile positions read" in the GOALS section, and to the card's "Ordinary text never contains the trigger". |
| 5 | Copy's gain masked off on both random ciphers, 0.84 and 0.92; identity control "mixed" at 0.31 | **Supported** | Scored by the filed rule. One addition: ident32 was P2's filed positive control (plan, P2). It missed its 0.25 line but still separates clearly from the ciphers: 0.31 against 0.84 / 0.92, and Γ's changed values carry 0.64 there against 0.04 / 0.13. "Goes the same way and a little further" than label training compares different steps (E10 at 30 and 10, the copy at 1,000; the copy at 100 reads 0.80 / 0.89). Say "the same way" and drop "a little further". |
| 6 | rand1 Pile-text copy "learnt": a reversal of the filed "neither" | **Supported as a reversal; the category is fragile** | P 0.9114 ≥ 0.9 and verdict pass (1.095 / 0.883) → "learnt" by the filed rule. 0.011 above the line, against a batch-order noise of 0.032 on Pile text. "Between" is as likely on a rerun, and it also contradicts "neither". The analysis says this. Allow: "the Pile-text copy reached 0.91 on the one-pair task (filed: below 0.5)". "Learnt the one-pair cipher" reads as more than a category name. What the copy can have learnt from text without the marker is the fine-tune's key-to-value bigram, which also answers the held-out windows. |
| 7 | Explanation of 6: the fine-tune shows the pair in plain text (0.31 vs 0.0001), "untested" | **Supported where hedged; overstated in two places** | The analysis hedges it in the answer, the section and Surprises. It overstates it in "On rand32 the Pile-text copy's gain of 4.1 log-odds … shows it picked up part of the leak". That is a causal claim with no test, and ident32's Pile-text copy gained 2.2 log-odds from a leak ratio of only 4.7, so the gain does not obviously follow the leak. The notebook overstates it in "The right panel shows a likely reason" and "Separately, the text-only copy can learn a new behaviour when the fine-tune shows it in ordinary text." Allow: "is consistent with it picking up part of the leak (untested)", and drop the notebook's general sentence. |
| 8 | "Two texts the project relies on disagree … this report does not pick" | **Overstated (unfair framing)** | See section 4. `GOALS.md` heads "Where an update goes" with "they do not decide the weak version", and the sentence right after the one quoted says the weak version is decided by VPD's checks. The files already rank the two readings. The open question is whether the user wants to change the criterion now that a case separates them. The 2026-10-19 decision says reading A is "open to change at some point in future". |

## 2. Number errors

Every table cell in the analysis matches the files: controls, the copy table, the trigger table, the route table, the
Pile-text table, the Pile reading, scorecard, provenance hashes and times, and the commit claims (`cef8e8f` for 40166,
`bb9b929` only moves the offsets save before the dump). The errors are in prose:

1. **"Γ's changed values carry … 0.64 to 0.72 on the identity control, for copies and do-nothing alike."** The
   ident32 Pile-text copy in the same table reads 0.288. Fix: "for the task-text copies, do-nothing and the
   least-squares split".
2. **"The residual is small here, 0.02 to 0.03."** The task-text copies at step 1,000 read 0.017, 0.030 and 0.034.
   Fix: "0.017 to 0.034".
3. **"Under the adversary it reads 0.96 to 0.99, inside the unchanged model's seed band"** (also in the notebook).
   Ratios are taken against W₀'s adversary seed 0 (`harness.pile_ratios`), which is the lowest of the three, at
   0.692 nats. On this scale W₀'s seeds read 1.00, 1.21 and 1.02, so the copies' 0.957 to 0.989 sit *below* the band,
   not inside it. Fix: "below all three of the unchanged model's adversary seeds".
4. **"so passes are lower bounds."** The adversarial *ratios* are lower bounds, since a stronger search can only find
   more. A pass therefore is not a lower bound; it may be lenient. Fix: "the adversarial ratios are lower bounds, so a
   pass may be lenient".
5. The marker count is cited for the wrong rows (claim 4 above).

Nothing else differs. Checked in particular: damage ratio 0.35 to 0.40, teacher gap under 1% of W₀'s (0.53%, 0.23%,
0.47%), CI-masked Pile ratios within 4% (lowest 0.969), Pile ratios 0.81 to 0.99, least-squares within 3% of 1.81
(0.7% and 2.3%), lines 1.24 / 1.81 and 1.497 / 1.503, Pile-reading ratios 146 / 3,065 / 4.7, masked-off margins 0.34 /
0.42 / 0.06, G1 and step-300 readings, sha256 prefixes of all four files.

## 3. Scoring and confounds

**Scoring.** C1 (`w0_check` pass, floor 0.2088, errors 0.25% / 0.05%), G1 ×3, power ×3, P1 ×3 pass, P2 masked-off /
masked-off / mixed, P3 neither / learnt / neither: each follows from the filed rule applied to the JSON, and the code's
`masked_class`, `p1_outcome` and `category` implement the filed rules as written (P2 boundaries inclusive at 0.5 and
0.25, P3's "learnt" needs P ≥ 0.9 and pass).

**L40 band.** It does not touch the pass verdicts: every copy reads 1.10 or less, under the H100's pass line too. It
only thins the power: the least-squares split fails by 0.7% and 2.3% on rand1 and ident32. On an H100 (fail line 1.61)
those would be clear fails, if the readings carried over. The copies sit 0.25 or more under the L40 pass line,
so their separation from the split is wide. The analysis states this fairly.

**One seed, one batch order, one adversary start.** Stated. It matters most for P3 on rand1 (claim 6) and for the
identity control's 0.31 against the 0.25 line. It matters least for the random ciphers' masked-off shares, which sit
0.34 and 0.42 over their line.

**rand1's G1.** As the analysis says, it cannot fail at tolerance 0.02, and step 300 is saturated too, so nothing
non-trivial ties the rand1 fine-tune to E10's run. The cost is small: nothing in the rand1 conclusions leans on E10
except the 0.86 label-trained comparison. The 4-decimal match (0.99824 against 0.9982) is weak evidence that the
windows file is the same one.

**0.911 vs 0.9.** Claim 6. The reversal stands whichever side of the line a rerun lands, since "between" also
contradicts "neither". The category "learnt" and the pass that comes with it should not be repeated as a finding.

**Not flagged by the analysis.**
- *Off-trigger, the copy is not the fine-tune.* Its damage elsewhere is 0.35 to 0.40 of the fine-tune's, and its
  Pile unmasked ratio is 0.81. Part of why the copy passes on Pile is that it moved the model less away from the
  trigger than the fine-tune did. The teacher gap's "under 1%" is dominated by the trigger term of W₀'s gap, so it
  does not show off-trigger fidelity. "Copies the fine-tune" should read "copies the fine-tune's behaviour at the
  trigger".
- *The stochastic ratio reads 0.87 to 0.94 for every state*, do-nothing and the least-squares split included. It
  separates nothing here and carries none of the verdicts. That is worth one line in "How compelling".

## 4. The draft pending-decision card

The card asks a real question and its Option 3 is a good one. It does not yet state the conflict fairly:

1. **The two texts are not equal.** The card says "the same file also says the weak version holds when [the
   masked-off share] is small". That sentence sits under a heading marking the section as "mechanism hypotheses …
   they do not decide the weak version (2026-10-19)". The very next sentence says the weak version is decided by VPD's
   checks. The plan stated this correctly ("and in the same section that VPD's checks decide it"). The card should say
   the criterion in force passes E42, and that the user's earlier mechanism sentence, now marked as not deciding, would
   fail it. Then ask whether this case is a reason to change the criterion, which the 2026-10-19 decision left open.
2. **"Passes every check the paper runs on ordinary text"**: the sparsity loss was not read, and there was one
   adversary start. Use claim 1's wording.
3. **"Ordinary text never contains the trigger"**: claim 4. Say "did not occur in the Pile rows scanned".
4. **"Every map tried behaves the same way at the trigger"**: true of the CI-masked probability, not of the reason
   (claim 3). Do-nothing and the least-squares split already fail reading A by putting the update in Δ.
5. **Option 2's consequence skips a cheaper branch.** "The commute line's next question becomes updating Γ" leaves
   out a frozen-Γ map that is already built but untried here: teacher training with VPD's masked losses
   (`masked_teacher.py`) on the task's windows. It trains the CI-masked and stochastic models toward the teacher at
   the trigger. Option 2 should say "updating Γ, or a frozen-Γ map trained against the masked losses on the task's
   inputs". It should also say that a task-input CI-masked check needs its own W₀ reference and line. E4's trigger
   ratios and E10's `retained_ci_masked` exist, and E10's label-trained states would fail it retroactively.
6. Urgency, run line and "why it matters" are fair. The $0 subcomponent list is recommended and cheap. Under the
   project's rule for cheap recommended checks, it should be run, not left on the card.

Suggested "Why it is a decision" text: *"The criterion in force (reading A, 2026-10-19, which the user said was open to
change) is decided by VPD's checks on Pile text, and by it E42 passes. `GOALS.md`'s mechanism section, which the file
marks as not deciding, says the weak version holds when the masked-off share is small, and by that E42 fails at 0.84
to 0.92 on the random ciphers. E42 is the first case where a map passes the first and clearly fails the second. Should
it change the criterion?"*

## 5. Wording I would allow into the log's sections above the cards

> Teacher training copied fine-tunes that teach the 67M model a word cipher it lacked (32 random pairs, one pair, and
> an identity control), with Γ and Δ frozen. On all three tasks the copy kept the behaviour (P 0.96 to 1.00) and
> passed the harness's reconstruction checks on Pile text; VPD's sparsity loss was not read. At the cipher's trigger,
> though, the model built from the subcomponents Γ marks important keeps almost none of it on the random ciphers
> (0.007 of 0.96, 0.001 of 1.00): the gain runs through subcomponents Γ keeps off, as with label-trained component
> training in E10, so this is not specific to teacher training. Whether any map with Γ frozen can avoid it is untested.
> Under the criterion in force this is a pass; whether the trigger reading should count is a pending decision. A copy
> trained on plain Pile text reached 0.91 on the one-pair task, against a filed "below 0.5"; the fine-tune shows that
> pair in plain text, untested as the cause.

## Not run

- No recomputation from the saved states (`results/large/e42/*/{A,B}_step1000.pt`); the trigger cells were taken from the logs as the analysis did.
- The marker count in the harness's 256 Pile validation rows.
- The figures and `plot_e42.py` were not opened.
- The E10, E34 Amendment 4 and 5 numbers the analysis cites from other reports were taken as filed, not re-read.
