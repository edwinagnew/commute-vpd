# Recheck of the corrections to the E34 Amendment 8 analysis

*2026-11-02. Checked commit d61a3d6 against the result files, filed predictions and saved offsets. The reviewer wrote none of it. Line
numbers are those of the report, the notebook file or `docs/research_log/PENDING.md` at d61a3d6. New $0 reading:
`~/commute-data/scratch/commute-lead/amend8-recheck/moved.py`, output `moved.out` beside it. Files below without a directory are in
`experiments/teacher_training/results/`.*

**Verdict in three plain sentences.** The numbers hold: every value in the headline, main table, scorecards and standing-claims table
recomputes, and all 16 review items were applied. The claim that how far a student moves barely depends on its teacher is wrong as
worded, because a smaller teacher gave a student 1.4 times larger, a one-factor change of teacher moved the student's move 1.7 times, and
the same task's students span a factor of 2 against about 2% for batch order; what holds is that a student's size does not follow its
teacher's size. Separately, the filed "within noise" is reported correctly where it is scored, but four sentences then lean past it, and
"five of six", "reproduces" and "a tenth of its move" need rewording before the seven queued log lines are flushed.

## 1. "How far a student moves barely depends on its teacher": wrong as worded

Where it appears: report lines 12-13, 117, 395; notebook line 55; `experiments/TRIED.md` E34 Amendment 8 row; `PENDING.md` lines 12 and 17.
The review's item 7 proposed "nearly independent", so the correction applied that item faithfully; the item itself went too far.

All rows: Adam 1e-5, batch 32, emoticon windows, 1,000 steps from the base unless stated. Size is the largest matrix.

| Teacher | Teacher size | Student size | Card | Source, json path |
|---|---|---|---|---|
| SGD fine-tune | 0.0122 | 0.0126 | H100 | `teacher_training_emoticon.json` `.teachers.sgd.nothing.size`, `.A["1000"].size` |
| Adam T10 | 0.0113 | 0.0141 | H100 | `teacher_training_e41.json` `.tasks.emoticon.teachers["10"].nothing.size`, `.A.oneshot["10"].size` |
| Adam T100 | 0.0222 | 0.0168 | H100 | same, `["100"]` |
| Adam T1000 | 0.0321 | 0.0184 | H100 | same, `["1000"]` |
| Adam T1000, regenerated | 0.0315 | 0.0182, 0.0184 at order 1 | L40 | `teacher_training_amend8c.json` `.real_teacher.size`, `.students.real_o0.size`, `.real_o1.size` |
| Null fine-tune, step 100 | 0.0277 | 0.0257, 0.0251 at order 1 | L40 | `teacher_training_amend8_n3b.json` `.reads.nothing.size`; `amend8c` `.students.null_o0.size`, `.null_o1.size` |

Five readings against the claim:

- **A smaller teacher, a larger student.** Null over real is 1.41 at order 0 and 1.36 at order 1, with the null teacher smaller (0.88 of the real one's
  largest matrix, 0.60 over all matrices: `amend8c` `.norms.null_teacher` 1.320, `.real_teacher` 2.203). Over all matrices the students read 1.225 and
  1.212 (`.norms`).
- **Far above batch order.** The same teacher at two batch orders differs by 1.0% (real) and 2.1% (null) in size, 0.2% and 0.9% over all matrices. The
  teacher's effect is about 20 to 40 times that noise. The filed `size` measurement puts batch-order spread under 1%.
- **The one-factor contrast.** From the same random start, the base as teacher moves 0.76 of the real student's offset norm, the null teacher 1.32, a
  ratio of 1.74 (`moved.out`; `cos.out` `rand_move`). The moved arm's move has cosine 0.97 with the null student's, so it is the same move from another
  start. Offsets, not δW.
- **The same task's spread.** Five teachers on emoticon give students from 0.0126 to 0.0257, a factor of 2.04, with the top on another card. Size is stable
  across cards here (T1000's student 0.0184 and 0.0182). That is close to the lowercase contrast called "follows": 0.0540 over 0.0236 is 2.29.
- **It depends on the step.** Null over real, order 0 then order 1 (`amend8c` `.curves.*[].rel_norm_max`): step 10, 0.78, 0.82; step 100, 1.36, 1.40; step 500,
  1.53, 1.48; step 1,000, 1.41, 1.36. The first few steps are alike. They separate after that, and the null student is the smaller at step 10.

What the data do support: size **does not follow the teacher's size**. Teachers spanning 2.85 times give students spanning 1.31 times along one fine-tune
(E41), and the null and SGD rows break any ordering by teacher size. The filed `size class` reads 1.41 as "about", but that class has no upper cap by design
and only means comparable; it does not mean independent. The null student ran only on the L40; no null student on an H100, on lowercase rows, or at a second label draw.

**"Not on lowercase rows" compares no teachers: overstated.** No two teachers were run on lowercase rows. 0.0236 and 0.0540 (`teacher_training_amend8_n4_adam.json`
`.reads["1000"].size`, `.reads.true.size`) are labelled training, with no teacher, batch 8, same job. 0.0365 (`experiments/teacher_spread/results/teacher_spread.json`
`.states.lower_s0.size`, H100) is the one student there, of a teacher of 0.0489, batch 8, plain mean KL. Each number is right. Read as "the teacher matters
there and not on emoticon" they do not show it, and 0.0365 / 0.0236 = 1.55 is hardly different from the 1.41 that the same paragraph calls barely depending.

**"Every Adam student ends within 0.76 to 1.41, however measured" (report 117, notebook 122): wrong.** Amendment 8b's moved arm is 0.0296 / 0.0182 = 1.62 in size and
2.16 in offset position. The random start is 1.88 in offset position (`cos.out` `rand1000`). E34's student of the SGD teacher is 0.68 on the H100. Two rows in the range,
self-labels and coefficient 0, are not students and not on this task.

### Replacements

- **Report lines 12-13.** "So a pass means "no harm seen". On this task a student's size does not follow its teacher's size, though it does differ by teacher, and
  about a tenth of its squared length lies along that teacher."
- **Notebook line 55-56.** The same sentence.
- **Report lines 117-122.** "**Reading it.** On this narrow task a student's size does not follow its teacher's size, but it does differ by teacher. At step 1,000 the
  null student reads 1.36 to 1.41 times the real one's, from a smaller teacher, where batch order moves either by about 2%. From one random start, changing only the
  teacher from the base to the null fine-tune takes the move from 0.76 to 1.32 of the real student's, in offsets. Across the five teachers read on this task, students
  run from 0.0126 to 0.0257 for teachers of 0.0113 to 0.0321, partly across cards. The gap opens with steps: null over real is about 0.8 at step 10 and 1.4 to 1.5 from
  step 100. On lowercase rows no two teachers were tried: labelled training reaches 0.054 on true labels and 0.0236 on self-labels, and E37's one student, another recipe
  on another card, reads 0.0365. So nothing here compares the teacher's effect between the two tasks. Self-labels at SGD reach 0.52 of the real student in offsets, and
  no arm varied the optimiser or the rate on the hook path. The random start's, round trip's and moved arm's sizes include their start."
- **Report line 395.** "On this task, a student's size does not follow its teacher's size, though it differs by teacher; no arm varied the optimiser or the learning rate on the hook path."
- **Notebook lines 122-124.** "**Reading it.** Not its teacher's size, on this task, though it differs by teacher. The null student reaches 1.4 times the real one's size from a smaller
  teacher, where batch order moves either by about 2%. On lowercase rows no two teachers were tried: labelled training reaches 0.054 on true labels and 0.024 on self-labels, and E37's
  one student, another recipe on another card, reads 0.037. No arm varied the optimiser or the rate on the hook path. The two walks back include their starts."
- **`experiments/TRIED.md`, E34 Amendment 8, last cell.** Replace "on this narrow task a student's size barely depends on its teacher (not on lowercase rows)" with "on this narrow
  task a student's size does not follow its teacher's size, though it differs by teacher, 1.36 to 1.41 null over real against about 2% for batch order; no two teachers were tried on lowercase rows".
- **`PENDING.md` line 12.** Replace "On this task size barely depends on the teacher;" with "On this task a student's size does not follow its teacher's size, though it differs by teacher (null over real 1.36 to 1.41, batch order about 2%);".
- **`PENDING.md` line 17.** Replace "on narrow tasks its size says little; on emoticon it is nearly independent of the teacher, and not on lowercase rows (E34 Amendment 8)." with "on narrow tasks
  its size says little about the update: on emoticon it does not follow the teacher's size, though it differs by teacher (E34 Amendments 8-8c, E41). No two teachers were tried on lowercase rows."
- **Figure title** `amend8_size.png`, from `plot_amend8.py`: the question now gets "yes, but not with the teacher's size". The title may stay; it needs only the new reading.

The standing-claims row "Add that on narrow tasks its size says little" (report 375) **holds**, on the E41 and Amendment 6 sizes.

## 2. The filed S1 and what is said beside it

**S1 itself: holds.** `teacher_training_amend8b_seeds.json` `.score.answer` "within noise"; `.score.null_real.excess` 0.3052, `.spread` 0.3667. KLs null 0.966, 1.144, 0.913; real
0.697, 0.935, 0.684. Rule text in report line 247 matches `predictions_amend8b.json` `.predictions[0].rule`. Report 12, 211, 404, `PENDING.md` 12, TRIED and the notebook all call it within noise and mark
the pairing "not the filed rule". Pairing 1.386, 1.224, 1.334 and 1 in 8: right.

The filed rule says 'within noise': no, the reading is not shown to be more than start noise. Four places restate it as an effect:

| Where | Wording | Verdict | Replacement |
|---|---|---|---|
| Report 60 | "Amendment 8b shows that this is not simply one start's reading" | Overstated | "Over three starts it averages 1.35, still inconclusive, and the filed test against the real student reads within noise." |
| Notebook 210 | "These allow that the reading is not one start's accident." | Overstated | "So the level holds over three starts, and the difference from the real student is not shown by the filed test." |
| Report 233-234 | "a student can read above the real student without any behaviour being taught, more than the size or the start alone accounts for" | Overstated | "a student can average above the base's level over three starts without any behaviour being taught, and sits apart from a similar-size student with the base as teacher; against the real student the filed test reads within noise." |
| Report 190, notebook heading | "Why does the null student read above the real one?", with "about twice the base's 0.21" under it, written before S1 | Overstated | Heading: "Why does the null student read higher at this start?"; add after that sentence: "Over three starts the filed test calls the difference from the real student within noise." |

**"The reading depends on what it copies": overstated in scope.** Report 235, 394, notebook 224, 250, TRIED 8b. Against the base as teacher it is supported: S3 "apart", filed; 0.975 to 1.56 from one start,
one adversary start. Against a real teacher, the filed answer is within noise. S3's own `blind_to` also says those two differ in size, damage, start and teacher together. Replace with: "So with the base as teacher against one
null teacher, the reading depends on what is copied; against a real teacher the filed test reads within noise." In TRIED 8b: "A student's reading differs between the base and one null teacher as what it copies; against the real student it is within noise."

## 3. Other lines that do not hold as worded

| Where | Wording | Verdict | Numbers, file | Replacement |
|---|---|---|---|---|
| Report 10; notebook | "Nothing breaks" | Overstated | Filed outcomes: N1 "fails, gap not closed" (`teacher_training_amend8_n1.json` `.score.outcome`), N2 "fails" (`n2` `.score.outcome`), N3b "inconclusive". The scorecard reports these; the headline hides them | "No run with the base as teacher fails the adversarial check." Report 369's "None breaks", of claims, stands |
| Report 57-59; notebook 106; `PENDING.md` 15; report 373 | "five of six", "12 of 13" | Stale after 8b, flatters | The moved arm is a seventh such run, 1.562 (`amend8b_moved.json` `.reads["1000"].pgd`); its two reads make 15. Both misses are the one null teacher | `PENDING.md` 15: replace "five of six runs with no behaviour to learn end at about 1, so a pass near 1 shows no harm and no more; a student of a teach-nothing fine-tune does not pass over three starts (1.35, inconclusive)" with "five of seven runs with no behaviour to learn end at about 1, so a pass near 1 shows no harm and no more; the other two are the students of one teach-nothing fine-tune, 1.35 over three starts and 1.56 at one, both inconclusive". Report 373 the same. Report 57-59 may keep its six if it adds "Amendment 8b adds a seventh, 1.56" |
| Report 13; notebook 55 | "about a tenth of its move lies along that teacher" | Imprecise | A share of squared length: 7.5% and 13.4% (`amend8c` `.score.along.*.share_own`). In length, the part along the teacher is 27-37% (`.cos_own`). "A tenth of its move" reads as length | "about a tenth of its squared length", as in §1 |
| `PENDING.md` 12, 15; report 373, 396 | "7-13% of a student's weight change" or "of its direction" | Imprecise | As above | "7-13% of the squared length of a student's weight change (cosine 0.27 to 0.37)" |
| Report 273-274; notebook 238; `PENDING.md` 12; TRIED 8c | "The other 87-93% ... reproduces, so it is not batch noise" | Overstated | Whole student's cosine between orders 0.927 and 0.974 (`.cos`), so on the report's own shared-vector reading 7% and 3% vary with batch order. Outside the teacher and reproducing: 0.927 − 0.075 = 85%, 0.974 − 0.134 = 84% of squared length | "The other 87-93% lies along neither teacher. Most of it reproduces across batch orders, about 84-85% of the squared length on a shared-vector reading, so it is mostly not batch noise." `PENDING.md` 12: "the other 87-93% is mostly reproducing and is unidentified". The report's plain "it reproduces" comes from the whole cosine, not a reading of this part |
| Report 271-272 | "that amount is about what its own teacher accounts for" | Loose | 10.5-11.1% against 7.5-7.9% for the real student (`.score.real_shared_with_null`, `.along_order0.real_o0.share_own`) | "that amount is more than its own teacher accounts for, 7.5 to 7.9%, though the quantities differ" |
| Report 110-111, notebook | "The whole gap on Pile is 0.012 nats" | Overstated | 0.012 is unmasked with Δ removed (`.W0.pile["0"].unmasked_kl`). The same state's CI-masked gap is 0.348, adversarial 0.692. The bound compares Pile with task windows. The conclusion survives at 0.348 | "The gap with every mask at 1 is 0.012 nats on Pile, and the CI-masked gap 0.348, against a start gap of 5 to 13 nats for a real teacher on task windows, so even if all of it were learned indirectly it could not be the update." The report's wording came from the review's item 10 |
| Card, report 354, option (b) | "If it reads above that at all three starts, repeating harms by itself" | Weak test | "That" is the mean 1.35. A chain at 1.40, 1.40, 1.40 satisfies it, and is inside the noise S1 just filed | "We read it against a single round at the same three adversary starts, 1.40, 1.37, 1.29. If the filed three-start test calls it apart from that round, repeating harms by itself ...; if within noise or below, repeating adds no harm we can see" |

## 4. Corrections that hold

**Review items, each applied and correct against the files.**
- Item 1, direction. Cosines own 0.281, 0.274, 0.366, 0.367; other 0.019, 0.047, 0.048; cross 0.086 to 0.091; within a teacher 0.927, 0.974 (`amend8c` `.cos`). `follows_own_teacher` true. Holds. "Differs between two teachers" is the strong, safe part.
- Item 2, shared share: 0.110, 0.111, 0.105 (`.score.real_shared_with_null`). Holds.
- Item 3, reproducibility, superseded by C1: `.score.reproduces_ratio` 1.050. Holds. Lean miss recorded.
- Item 4, jolt: clause gone; base-teacher jump 0.0070 to 0.30 (`amend8b_moved` control 0.0070 to 0.302). Logs of E41 Amendment 2 were not reread here; the first check read them.
- Item 5: 0.0141 to 0.0214 is 52%, 0.0214 to 0.0251 is 17% (`teacher_training_e41.json` `.tasks.emoticon.A`); round trip 17.2%, random start 25.1%. Holds.
- Item 6: 0.0269 − 0.0180 = 0.009; 0.0460 − 0.0363 = 0.010. Holds (values as the first check read them from `teacher_training_e41.json`).
- Item 8: return −0.172, −0.251 (`.score.return_fraction`); cosine 0.6416, length 0.6416 × 1.184 / 1.027 = 0.74, squared −0.33 (`cos.out`). Holds. The card's "about a quarter of the original direction was undone" omits "in offsets"; the report's section says so.
- Item 9: wording applied; now stale in one count (§3).
- Item 10: fall 0.945 against 0.821 one-factor, 0.821 from the base. Holds, except "whole gap" (§3). "A third to a half" is, on the hook path, 0.44 from one run, and coefficient 0 reads 0.75; marked rough, and fine as that.
- Item 11, the do-nothing row: 2.636 to 1.391. Holds.
- Item 12: card states the card and the branches; one test weak (§3).
- Item 13: recommendation wording. Holds.
- Item 14: 0.122, 0.120; 2.64, 1.39. Holds.
- Item 15: `.score.students_inflation` "not shown", 0.0019; class fragility stated. Holds.
- Item 16: null curve KL 0.519 to 0.579 from step 30 (`n3a` `.null_checkpoints`, 0.587 at step 3); others applied. Holds.

**Every number recomputed.**
- Main table, lines 65-84: all 20 rows against `teacher_training_amend8_{n1,n2,n3b,n4_adam,n4_sgd,n5}.json` `.reads` and `amend8b_moved` `.reads`.
- Counts: 13 reads, 12 under the band top 1.208, 9 below 1 between 0.923 and 0.999.
- Means 1.353, 1.032, 0.946, 7.553 (`amend8b_seeds` `.score.mean_ratio`); S3 "apart", excess 0.435, spread 0.254.
- M1: 0.01154, 0.00970, 0.04258, position 0.117 (`amend8b_moved` `.score`). The matching of damage, stochastic, CI-masked, Δ-out and train KL: holds.
- C1 to C3 as scored. M0 25.99 (`teacher_training_e42_amend1_rand32.json` `.M0.pgd`). E37 0.0365, 0.0489 (`teacher_spread.json`).
- Offset ratios 0.76, 0.92, 1.21, 1.30, 0.52 (`cos.out`).
- Standing-claims table numbers: all recompute. Their verdicts are not changed by this recheck except the wording rows above.
- Scorecard rule text for S1 to S3 and M1, C1, C3 against the predictions files: verbatim.

**`PENDING.md` lines that hold unchanged:** 13 (decision), 14 (TRIED wording), 16 (one start stated), 18 (17% and 25% against 17% and 52%; 1.28, 1.31, 1.20; 0.98).

**`plot_amend8.py`:** the new figures read the right keys; `amend8b_seeds.png` matches the JSON. Holds. Bf16 reads there (1.40, 1.01) differ from fp32 (1.39, 0.99) inside the filed 7%.

## Not checked

E41 Amendment 2's logged train losses (first check read them); E39's 1.27 and 2.12 in line 238; figures other than `amend8_size` and `amend8b_seeds`; the full E34 notebook outside these cells. `check.py` on the corrected files was not rerun.
The moved arm's largest-matrix move, as opposed to its position, was not read; `moved.out` is offset norms over all alive columns.
