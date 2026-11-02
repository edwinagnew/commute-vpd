# Independent check of the E34 Amendment 9 analysis and code

*2026-11-02. Checked at commit 30a93703. The reviewer wrote neither the code nor the analysis. Line numbers are those of
`reports/analyses/2026-11-02_analysis_e34-amend9-student-move.md` unless another file is named. New $0 reading, no model, no GPU:
`~/commute-data/scratch/commute-lead/e34a9-check/recheck.py`, with `recheck.log` and `recheck.json` beside it.*

**Verdict.** The code and every number are right: all ten students recompute from my own projectors, and I found no bug.
Three readings overreach: "taught nothing", the reach reading "two different things", and "no support" for copying, which rests on a line that a rerun of the same fine-tune also fails.

## What is right

- **Numbers.** Every value in the report's tables, figures and scorecard agrees with `amend9.json`. I recomputed all ten students and six fine-tunes without `split.py`, `offsets_state` or `change_stats`: own thin SVD of $U$ and $V$, own rebuild of $s$ in fp64, own inner products. `cos_all`, `cos_in`, share, size, own inside share and overlap outside agree to at least 9 decimals. Reach is correctly cosine times size; it equals $\langle s, I\rangle / \lVert I\rVert^2$, recomputed directly.
- **Projector.** $O$ matches $(I-P_U)\,\cdot\,(I-P_V)$ built independently. The isotropic reference `gaussian_null` is the right formula, weighted by per-matrix energy.
- **Alive set.** All 24 matrices are full column rank, or saturated on one side. E12's students re-read their filed cosines within 4.3e-7 on `uv0.pt`, so `uv0.pt` is a sound stand-in. A wrong $U, V$ would change $s$.
- **Layouts.** On the 16 square matrices, a student against the inside part reads 0.33 to 0.52, and against its transpose at most 0.0013 in size. Swapping $U$ and $V$ lowers the square matrices' share from 0.42 to 0.37 on the real fine-tune and from 0.44 to 0.41 on the null. The layout is right, and C1 and C2 could not have shown this: they hold under any layout.
- **Pairings.** Read from the training code. E12's students trained against exactly `regating_table/results/large/{task}_{seed}.pt` (`new_vs_existing/amend2.py:203,231`); 8c's real student against the regenerated T1000 at Adam 3e-5 (`amend8.py:345-366`), its null against step 100 (`amend8c.py:206-211`). `amend9.py:212,215` and the `own` map at line 110 match. "Second table" is rand32b, seeds and orders are right, and the figures' y-labels line up with their points. Scored steps are 10 (rand32), 30 (rand32b), 100 and 1,000, so line 108 is right. All students share one recipe: Adam 1e-5, batch 32, 1,000 steps.
- **Rules and scoring.** P1, P2 and P3 are scored as filed, and the `score` code matches the filing. The miss on P2 read 2 is reported and not refiled. Predictions sha matches, filed in 88a98bbf before `fe676ce6`; those commits are two seconds apart, both before the first logged line (18:19:05). The job took about 2 minutes. Importing E12's `amend2.py` rebinds `amend1.OUT`, but nothing here calls `amend1.dump`, so that side effect is harmless.
- **The caveats that are stated.** Input-weighting, pooling and the unit of one label draw are all stated, and the report does not claim a cause for the null gap or for reach.

## Findings, most important first

### 1. "Taught nothing" and "does not need a behaviour" go past the data (lines 8, 39, 83, 124; `plot_amend9.py:30`; notebook cells 49, 51, 53)

**Failure.** The null fine-tune teaches no task, but it moves predictions 0.53 nats, and Amendment 8's report says of the same state "So its student has something to copy" (`2026-11-02_analysis_e34-amend8-base-control.md:33`). Line 85 repeats the 0.53 and then line 83 says "a high value does not need a behaviour to learn". The student was trained to reproduce a real change in outputs. What the null removes is a *task*, not a behaviour.

**The comparison is also framed one way.** The null does not read above the taught tasks. It sits inside them: first table 0.365 to 0.392, null 0.461 and 0.462, second table 0.481 to 0.506. Both map tables learned their task (held-out P(target) above 0.5). The real-against-null gap of 0.10 to 0.11 is the same size as the gap between the two map tables, 0.09 to 0.14. Line 79's "four times the largest batch-order change" therefore uses the wrong noise: batch order says the gap is not rounding or order, but between-task spread is the relevant yardstick for what the gap means. With one label draw, nothing here shows where a second null would read.

**The conclusion survives, on better ground.** The cosine cannot measure teaching, because taught tasks and the null overlap. Line 85's list of untested causes should also carry the gap: the null student ends 0.0097 nats from its fine-tuned model and the real student 0.0286, so the cosine may go with how closely outputs were reproduced.

**Corrected wording, line 8.** "...on the real task about an eighth of the squared weight change lies along that part. A fine-tuned model trained on labels drawn from the unchanged model's own predictions, which teaches no task but still moves predictions 0.53 nats, reads about a fifth, inside the range of the two map tables that did learn a task (0.13 to 0.26), so this cosine cannot be read as how much was taught."

**Corrected wording, line 83.** "Tasks taught and a fine-tune that taught no task read in the same range, so a high value does not need a task. It may still need an output change to reproduce: the null student had one." Change the legend to "null fine-tune, no task". Say at line 8 that the comparison is unfiled, as line 81 already does.

### 2. The reach split is largely the fine-tune's length, so "two different things" overreaches (lines 12, 104, 106, 108, 140; notebook cell 52)

**Failure.** Reach is a length over the fine-tune's length, and the report reads it from the student's side ("moved a third of that length"). In absolute weight length, from my recompute:

| Family | Steps | Inside part's length | Student's length | Student's length along the inside part | Reach |
|---|---|---|---|---|---|
| Real | 1,000 | 1.73 | 1.60 | 0.56, 0.57 | 0.32, 0.33 |
| Null | 100 | 1.05 | 1.94, 1.96 | 0.90 | 0.86 |
| First table | 10 | 0.61, 0.62 | 1.71, 1.73 | 0.63, 0.67 | 1.02, 1.10 |
| Second table | 30 | 1.07, 1.09 | 2.02 to 2.03 | 0.98 to 1.03 | 0.90 to 0.96 |

The first table's students and the real students differ by a factor of about 3.2 in reach and only about 1.15 in length along the inside part. Their cosines overlap too (0.365 to 0.392 against 0.348 to 0.357). So that split is all denominator: 2.8 times the inside part's length. Against the null and the second table it is about half and half: the real student's length along the inside part is about 0.6 of theirs, and its fine-tune is about 1.6 times as long. So line 106's "a low cosine means two different things" does not follow: in every family the part at right angles is 1.7 to 2.7 times the part along. What differs most is the student's size against the fine-tune, and students' sizes do not follow their fine-tunes' (1.60 to 2.03 against 0.61 to 1.73), as the Amendment 8 recheck also found.

**Reach is fairly defined but not new.** The definition was filed and line 32 is correct, and line 108 is honest that no rule reads it. Line 60 lists it among things "new" with "no earlier reading", while lines 108 and 140 say it was implied. Nearly so: $\text{reach} \approx \cos(s, \delta W) \times \text{size} / \text{share}$, and only the share was unknown. For the null the filing gave no size range, so "implied by the filed ranges" (line 140) does not cover it.

**Smaller errors in the same passage.**
- Line 104's "on or just under it, 0.86 to 1.10": the first table is above the curve (1.02, 1.10).
- "Reaches about all" and the figure's legend "reaches all of it" read as containing the inside part. It is one projection, and that sits badly beside line 126's "no support". Line 104 should say so.
- Pooled reach hides a block split. On the 4 saturated matrices it reads 1.00 to 1.26 on the three short families, on the other 20 it reads 0.70 to 0.90. The real student reads 0.38 to 0.39 and 0.26. The split between real and the rest holds in both blocks.

**Corrected wording, line 106.** "A student's length along the inside part (0.56 to 1.03) does not follow the inside part's own length (0.61 to 1.73). So reach depends heavily on how long the fine-tune was: the real student's length along it is close to the first table's, and its reach is a third of theirs because its fine-tune is 2.8 times as long. Against the null and second table, half of the difference is that length and half is the student's own shorter move along it. Reach 1 means one projection matches in length, alongside a part at right angles about twice as long; it does not mean the student contains the inside part."

**Line 145 can be answered now.** Per matrix, the real student's reach has median about 0.3 and runs from −0.09 to 0.50, so no one matrix carries it. Other families' medians are 0.76 to 1.02, ranging 0.42 to 1.49. No matrix in any student crosses the filed line: largest per-matrix cosine 0.654. One oddity for whoever follows up: `h.3.attn.k_proj` reads cosine −0.10 and −0.13 on the real students, at both orders, where the null reads +0.56. It holds 1.2% of the student's squared length and is unexplained.

### 3. The line at 0.71 fails a rerun of the same fine-tune, so "no support" for copying is too strong (lines 8, 77, 126)

**Failure.** The controls are the least-squares split (1) and the do-nothing (0.79 to 0.82), which show the apparatus works but not what a same-process rerun reads. I read it: the inside parts of two fine-tunes of one table at two seeds have cosine **0.63 on rand32 and 0.70 on rand32b**, both under 0.71. So a second fine-tune of the same task would itself read "mostly elsewhere". This does not change P1, which is scored as filed, and students' 0.37 to 0.51 sit well below the 0.63 to 0.70. It does change what the verdict can say. The caveat: these fine-tunes are 10 and 30 steps. The real task has no second fine-tune saved, and what a rerun reads at 1,000 steps is unknown.

**Corrected wording, line 126.** "'Copies the fine-tune into the subcomponents' is not supported by weights counted alike: map-table students read 0.37 to 0.51 against 0.63 to 0.70 for the same table's fine-tune at another seed. But the filed line at 0.71 is one that this rerun fails too, so the reading is 'less aligned than a rerun', not 'elsewhere'. That read cannot speak to the directions inputs excite." Add "a second real fine-tune" to Not run; it needs training.

### 4. "Stays inside it" names the wrong thing (line 8; notebook cell 49)

**Failure.** In the opening, "that part" and "it" are $I(\delta W)$, one fixed direction. The own inside share reads $\lVert I(s)\rVert^2/\lVert s\rVert^2$, membership of the *space* of changes the subcomponents can write. The sentence before says the student's squared cosine to $I(\delta W)$ is an eighth, so as written it contradicts itself. It is also the one fact the report later calls true by construction (line 126), and should not be listed as a check the run added without saying so.

**Corrected wording.** "...plus the checks that each weight change stays within what the subcomponents can write, as it must to first order, and follows its own task's fine-tune far above another task's."

### 5. "Two-thirds of every update" hides four saturated matrices (lines 8, 58, 124; notebook cells 50, 53)

**Failure.** `h.0` and `h.3`'s `c_fc` and `down_proj` are saturated on one side, so everything in them is inside by construction, for any update. They hold 32 to 37% of each update's squared length and **51 to 56% of the inside part**. On the other 20 matrices the share is 0.44 to 0.47, against 0.40 to 0.43 for a random update. The filing knew this (`predictions_amend9.json`, `what_is_settled_before_the_run`); the report never says it. "Barely more than a random update" holds; all of the excess is on the 20, where it reads 0.03 to 0.05. "The subcomponents could write two-thirds" reads as a property of the update and the subcomponents' fit when a large part is a matter of which matrices are full rank. Line 131's "the larger matrices dominate" does not cover it. For the real fine-tune "two-thirds" is also generous: 0.62.

**Corrected wording, line 58.** "About 0.62 to 0.67, barely more than a random update. Four of the 24 matrices are saturated, so anything in them can be written; they give over half of the inside part. On the other 20 the share is 0.44 to 0.47, against 0.40 to 0.43 for a random update."

### 6. "Which fine-tune" is really "which task" (lines 8, 77, 83, 124; notebook cells 51, 53)

**Failure.** The "another" read so far is another task's fine-tune, and those fine-tunes are themselves nearly orthogonal (inside parts 0.05 to 0.07 across tables, 0.06 real against null), which gives most of the open marks: 0.39 × 0.068 is about the 0.028 read, though the null student's 0.060 on the real fine-tune is about twice what its 0.063 overlap would give. I read the same table's other seed, which line 186 lists as not run. Own against other seed: 0.392/0.299, 0.365/0.310, 0.505/0.411, 0.487/0.437, 0.481/0.443, 0.506/0.421. So a student's own fine-tune does read higher in all six, by 0.04 to 0.09, 1.5 to 4 times the largest batch-order change, 0.025, against a gap of 0.33 or more across tables.

**Corrected wording, line 124.** "The cosine tells which task's fine-tune a student followed, and only weakly which seed's." Line 77: "so the number does follow the student's own task." Move "another seed's fine-tune" out of line 186 and into the report, citing this review.

### 7. Holding back the filed consequence is defensible, but its reason should be stated as a flaw in the filing (lines 10, 126)

**Failure.** The disclosure is honest, and the verdict is unchanged. But line 10 gives only the input-weighting reason, while line 126 gives the stronger one: "a change the pieces can express" is true of any student by construction. So the filed fork was not between two live wordings. "Copies" was nearly excluded by arithmetic, and "expressible change" could not fail. The plan (line 17) says "This run decides between two wordings" and line 29 says P1 "picks the wording". A reader of line 10 cannot see that. Setting aside a filed consequence also affects the commute line's headline wording, so it belongs in the result message rather than only here.

**Corrected wording, line 10, added sentence.** "The filed choice was also weaker than it looked: the second wording's 'can express' holds for any student by construction, so P1 could only say 'not copies, counted by weight'."

### 8. Small items

- Line 60, "rescaled by the share": it is rescaled by the square root of the share, as line 48 has it.
- Line 60, "at most 0.0003": correct, but small only because the outside part of $s$ is tiny. Within it, $O(s)$ has cosine 0.16 to 0.27 with $O(\delta W)$ on the short families and 0.01 to 0.02 on the real one. Not a failure of the identity, and nothing should rest on it.
- Line 25, "in fp64": inner products are, but `offsets_state` stores $s$ in fp32. My fp64 rebuild agrees to 9 decimals, so no number moves.
- Line 104, "as they must": exact only to first order. The figure plots the whole state, whose product is up to 1.0017 (0.99970 × 1.0020 on the null).
- Notebook cell 53, "The null student above the real one was filed": only a range for each was filed. Line 81 has it right.
- The E34 line of `docs/research_log/PENDING.md` (line 10) and the notebook carry the opening's wording, so findings 1, 4 and 6 apply there too when the queue is flushed.

## What I did not check

Whether the saved fine-tune files still match the sha256s in `stage0a.json` (the filed cosines reproducing is the evidence I relied on); the fine-tunes' and students' training itself; anything input-weighted, which needs a forward pass. Line 130's 0.0286 nats I confirmed against `teacher_training_amend8c.json` (`students.real_o0.gap`) but did not recompute.

## What the lead did

*2026-11-02. Applied in the report, notebook cells 49-53 (no re-execution), `plot_amend9.py` (figures regenerated from `amend9.json`), the TRIED row and the PENDING line. Numbers and filed verdicts unchanged; `predictions_amend9.json` not edited. The reviewer's new readings are marked "independent check", unfiled.*

1. "Taught nothing": reworded to "teaches no task but still moves predictions 0.53 nats", the null placed inside the map tables' range, the 0.0097 against 0.0286 nats gap added to the untested causes, the legend changed to "null fine-tune, no task", the comparison marked unfiled. Applied.
2. Reach: the "two different things" reading replaced by the reviewer's length wording, the absolute-length table, the saturated/other block split, the per-matrix reads and the `h.3.attn.k_proj` oddity (open, no cause) added; reach no longer called new; "reaches all of it" changed in text and legend. Applied.
3. The 0.71 line failing a rerun: the line-126 wording applied with the 0.63 and 0.70 rerun cosines; "a second real fine-tune" added to Not run. Applied.
4. "Stays inside it": reworded to "stays within what the subcomponents can write, as it must to first order", and the own inside share is no longer listed as a check the run added. Applied.
5. "Two-thirds": line-58 wording applied, with the saturated-matrix split and 51 to 56% of the inside part. Applied.
6. "Which fine-tune": reworded to "which task's", the same-table other-seed read added, and removed from Not run. Applied.
7. Filing flaw: stated in the report opening, the scorecard and the "What we learned" section, and in the notebook; not a new verdict. Applied.
8. Small items: square root of the share, the $O(s)$ cosines, fp32 storage of $s$, first-order "as they must", and the filed range for the null; all applied.
