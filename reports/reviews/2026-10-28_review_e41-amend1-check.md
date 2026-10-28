# E41 Amendment 1: independent check of the analysis section and its card (review)

*2026-10-28, independent agent that did not write the code or the analysis; read-only apart from this file. Checked
from the raw files: `experiments/teacher_training/results/teacher_training_e41_amend1.json`, the run log (now at
`experiments/teacher_training/results/large/run_e41_amend1.log`, see § 5), E41's results and log for order 0,
`predictions_e41_amend1.json` (committed once, d95a037 at 13:26:03, before the job's first log line at 13:37:50),
`experiments/teacher_training/path_amend1.py` (`near_line`, `score`, `LINES`), `plot_e41_amend1.py`, E34 Amendment 4's
`scorecard_amend4.json`, `decisions.md`, the plan's "Amendment 1", the analysis section "Amendment 1" with its card
draft, and the notebook's Amendment 1 cells. Arithmetic done by hand from the stored values (no Python was needed).
The three sha256 values in the analysis's provenance match the files.*

Every number in the section's tables, figure and scorecard matches the results file and the log, and both predictions
are scored by their filed rules. The section is careful about what it measured; it goes too far in what it attributes
the spread to. Each reading had one adversary start, so "batch order moved the reading by a sixth" should read "the
reading varied by up to a sixth across batch orders"; the copies are near-identical on every number the adversary does
not touch. The word "damage" is used for the check's excess, though the measured damage of every copy equals the
teacher's. The card is fair in shape but needs four repairs: its 160-step adversary does not separate the two noise
sources; "works on fine-tunes run to the end" ignores E37; the recommendation's reason assumes the cause; and the cost
of (e) on the short-links run it blocks is missing.

## 1. Numbers

Pass and fail lines at this job's floor 0.0748164: 1.5 / 1.0748164 = **1.39559** and 1.5 × 1.0748164 = **1.61222**.

| Quantity (analysis) | Raw value | Check |
|---|---|---|
| C1 floor; PGD at adversary seeds 0-2 | 0.074816; 0.69882 / 0.75110 / 0.70672 | match |
| W₀ seed-0 blocks | 0.6794 / 0.7182 (log lines 6-7) | match |
| W₀ block ranges across starts, "0.99 to 1.06" and "0.96 to 1.15" | block 1: 0.6745, 0.7230 over 0.6794 = 0.993, 1.064; block 2: 0.6904, 0.8277 over 0.7182 = 0.961, 1.152 | match |
| G1 | 0.999312 / 0.011254; 0.999611 / 0.032090 | match |
| G2 | P 0.997901, ratio 0.990066; 3.2% from E41's 0.95909 | match (0.99007 / 0.95909 = 1.0323) |
| do-nothing | 2.5159, blocks 1.1685 / 3.7906, stochastic 1.0026 | match |
| orders 0 / 1 / 2: ratio | 1.54653 / 1.27967 / 1.50543 | match |
| verdicts | inconclusive / pass / inconclusive against 1.39559 and 1.61222 | match |
| order 1 "a pass by 8.3%"; order 2 "6.6% under the fail line" | 1 − 1.27967 / 1.39559 = 0.0831; 1 − 1.50543 / 1.61222 = 0.0662 | match |
| order 0 blocks 1.06 / 2.00 | 0.7218 / 0.6794 = 1.0624, 1.4397 / 0.7182 = 2.0046 (E41 log lines 164-165) | match |
| orders 1, 2 blocks | 1.0742 / 1.4741; 1.0652 / 1.9219 | match |
| excess removed 0.64 / 0.82 / 0.67 | 0.6395 / 0.8155 / 0.6666 | match |
| P(target), gap share, stochastic, size, damage, start KL, P at 100, Δ share | all three rows as in the results files | match |
| −17.3%, −2.7%, max/min − 1 = 21% | 0.82745, 0.97342, 0.20854 | match |
| "half as large again", "five times emoticon's" | 17.3 / 11.1 = 1.56; 17.3 / 3.2 = 5.4 | match |
| E34 Amendment 4 task-text gaps "up to 11.1%", emoticon "3.2% and 0.1%" | arm A: r0 Adam +11.10%, emoticon Adam +3.23%, emoticon SGD +0.10% | match |
| card: E37 from the adjoint split 1.05 then 1.76; longer training 2.20 / 1.89 / 2.23 | research log E37 card | match |

Number and wording errors:

1. **"Two-thirds or more" (card), "two-thirds to four-fifths" (What we learned, notebook).** Order 0 removes 0.64,
   under two-thirds. Write "64% to 82%" or "about two-thirds to four-fifths".
2. **"Damage" for the check's excess** (The answer, the card's evidence paragraph, the heading "Where is the extra
   damage?", the notebook's "extra damage"). `damage` is a measured column, and every copy's is 1.02 to 1.04 times the
   teacher's. So "removes most of the damage that leaving the update in Δ causes" contradicts the section's own table.
   Write "removes 64% to 82% of the excess that leaving the update in Δ shows on the adversarial check".
3. **"A quarter to a third larger than the teacher."** Sizes are 1.24, 1.25 and 1.30 times the teacher's. Write "a
   quarter to 30% larger".
4. **Notebook: "The second block carries all the excess."** The first block reads 1.06 to 1.07, so the second carries
   86% to 94% of the excess over 1. Write "nearly all".
5. **"That margin is outside the 7.5% adversary-seed floor."** True, by 0.8 points, and the floor is W₀'s spread over
   starts, not this copy's. Write "just outside W₀'s 7.5% spread across adversary starts".

## 2. Predictions, scored by the filed rules

- **P1, inconclusive (lean inconclusive): correct.** No order loses the behaviour: P is at least 0.9990 and the gap share
  at most 0.0014, far inside 0.9 and 0.05. Verdicts are inconclusive, pass and inconclusive, so the median is
  inconclusive. The counts are 1 pass and 2 inconclusive, and the run is not split. `near_line` uses |ratio / line − 1|:
  order 0 is 4.1% from the fail line, order 1 is 8.3% from the pass line (not flagged), order 2 is 6.6% from the fail
  line and 7.9% from the pass line. The flags "fail, none, fail" match.
- **P2, fails (lean holds): correct.** The band is 1.37481 to 1.71825. Order 1's 1.27967 is below it and order 2's
  1.50543 is inside. The rule needs both, so P2 fails. The comparison with 11.1% uses the same convention (signed, over
  order 0) as `scorecard_amend4.json`, so the filed rule is like for like in arithmetic.

## 3. Verdict per claim

| # | Claim (section wording) | Verdict | Why, and the wording I would allow |
|---|---|---|---|
| 1 | Opening: "Batch order moved this copy's reading further than it has moved any copy of a finished fine-tune" | **Overstated** | Each state was read with one adversary start, so the spread mixes batch order with the adversary's own variation. Allow: "Its reading varied more across batch orders than any finished copy's has. Each reading used one adversary start, so part of that may be the adversary's." |
| 2 | The answer: "One change of batch order moved this copy's reading by a sixth, where it moved a copy of the finished fine-tune by a thirtieth" | **Overstated and selective** | The numbers are right, but the other new order moved 2.7%. The finished-copy reference is up to 11% across tasks, and the 3% is one pair. Allow: "Across three orders its reading spans a fifth; one new order read 17% below the first. At a second order, finished copies from task text moved at most 11%, and this fine-tune's 3%. With one adversary start per reading, how much of this is batch order is untested." |
| 3 | The answer and card: "removes most of the damage that leaving the update in Δ causes" | **Wrong word** | See § 1 item 2. Allow: "removes 64% to 82% of the excess that leaving the update in Δ shows on the adversarial check". |
| 4 | Controls: all passed, reproduced to the digit; the seed was applied (start KL 4.986, 4.962 against 5.427) | **Supported** | G2's exact reproduction of Amendment 4's order-1 copy is the stronger evidence that the seed was applied. |
| 5 | "At one order of three. ... No order fails, and no copy loses the behaviour." P1 inconclusive, as leaned | **Supported** | |
| 6 | "Does batch order move the step-10 copy more ...? Yes." with "The 11.1% reference was measured the same way, so the comparison is like for like" | **P2 fails as scored; "Yes" and "like for like" overstated** | The procedure is the same, but the adversary's variation may not be. The finished copies read near 1, where W₀'s spread across starts is 7.5%. This copy carries a large excess on one block, and W₀'s second block alone spans 20% across starts. Whether the adversary's variation grows with the excess is unmeasured. The copies agree on every number the adversary does not touch: stochastic ratio 0.979 to 0.981, damage 0.057 to 0.058, gap share 0.0013 to 0.0014, size 0.014 to 0.015, first block 1.06 to 1.07. They differ only in what the adversary finds on the second block, 1.47 to 2.00. Allow: "By the filed rule, yes: one new order read 17% below the first, outside the 11% filed. The copies agree on every number except what the adversary finds on the second 128 rows, and each got one adversary start. So whether the copy or the adversary's start moved is untested." |
| 7 | "Where is the extra damage? In the second block ... every order leaves a real excess on the same 128 rows, and batch order changes its size" | **First half supported; second overstated** | 1.47 to 2.00 against W₀'s 0.96 to 1.15 there is a real excess at every order. Allow: "...and its size varies across orders, 1.47 to 2.00." Heading: "Where is the excess?" |
| 8 | What we learned: "teacher training from task text sits at the pass line" on the earliest update of this fine-tune | **Supported** | It is scoped to this fine-tune, which is right. |
| 9 | What we learned: "Its reading is more sensitive to batch order than any copy of a finished fine-tune." Surprises: "more sensitive to batch order than any copy of a finished fine-tune measured so far" | **Overstated** | As claim 1. Allow: "Its reading varied more across batch orders than any finished copy's, with the adversary's start not separated." |
| 10 | What we learned: "small updates are where teacher training is least certain to keep the decomposition faithful" | **Overstated** | This rests on one step, step 10. Emoticon was read at three orders and " which" at one; " which" passed at 1.376, 1.4% inside its line and itself within the adversary-seed floor. At step 100 doing nothing passes, so the check has no power there. Allow: "On both tasks the step-10 copies read closest to the pass line of any copy. On emoticon, three batch orders straddle it." |
| 11 | What we learned: the adjoint split "passed at 1.06 in E22, though it loses the behaviour later" | **Supported, incomplete** | Add its P 0.93 at that step (copy 0.999), since the card's option (d) leans on it. |
| 12 | How compelling, Scorecard, Not run | **Supported** | Not run could add "a second set of Pile rows", which is named under How compelling. |
| 13 | What next: the tree sends this outcome to a Blocking card | **Supported** | The plan's tree for "inconclusive" lists more starts, the budget adversary or accepting the reading, and for "P2 fails" asks for the spread in the card. The card does both and adds three more options. |

## 4. The card as I would allow it

Changes, in order:

- **Title.** "least certain on the earliest, smallest updates" rests on one step. Allow: "On the step-10 emoticon update,
  teacher training reads at the pass line: accept that, or try to fix it?"
- **"It works on fine-tunes run to the end."** It passed on the six trigger-task fine-tunes (E34), but not on the broad
  lowercase fine-tune (E37, 2.20). Scope it to the trigger tasks.
- **Evidence paragraph.** Fix "damage" and "two-thirds or more" (§ 1). Add the start caveat to "by a sixth". Give
  " which" its margin. Give the adjoint split's P.
- **(a) "No cost."** True in dollars, but it leaves the run line's own question open: under (a) alone the short-links
  run reads its early links at one order. Say so.
- **(b) conflates two things.** Extra adversary starts on the same copy do separate the adversary's variation from batch
  order. The 160-step adversary does not. It is the stronger search `decisions.md` set aside on 2026-10-27, and on W₀
  it varied more across starts at 160 steps than at 20 (1.23 against 1.07, E34 Amendment 4's C1b). Keep (b) as
  starts only. Name it a noise measurement: each start reported, the verdict still read at seed 0, so it is not the
  "search over several starts" the decision rules out. It is still a departure the user should approve. Its cost is
  about $2: three copies regenerated with six extra reads, roughly 16 minutes of job plus set-up. Add what it buys.
  It is the only option that says whether the standard in (e) should be three batch orders or three adversary starts
  on one copy. Here an extra read costs about 1 minute and an extra copy about 2. On Qwen, copy training dominates, so
  the gap widens.
- **(e) cost.** Add its cost on the run the card blocks. Three orders triple the approved short-links chain, about
  15 minutes and $1.5 in its plan, to about 45 minutes and roughly $4 (set-up is not tripled).
- **Recommendation.** "(a) with (e)" is defensible: a median over three orders is a robust reading whichever noise
  causes the spread. But its reason, "one order alone misread it by 17%", assumes that batch order is the cause and
  that the median is the truth. Allow the reason as worded below, and name (b) as the cheaper standard's test.

> **On the step-10 emoticon update, teacher training reads at the pass line: accept that, or try to fix it?**
>
> **What it is about.** `GOALS.md` assumes post-training updates are small. Teacher training copies a fine-tune into
> the subcomponents with Δ frozen. It passes on the six trigger-task fine-tunes run to the end (E34), though not on
> the broad lowercase fine-tune (E37). We asked whether it also works on a fine-tune stopped after 10 steps. That is
> the one point where leaving the update in Δ fails the adversarial check (2.52).
>
> **What the evidence says.** On emoticon, three copies trained at different batch orders read 1.55, 1.28 and 1.51.
> Passing means 1.40 or less and failing 1.61 or more, so one passes, two are inconclusive and none fails. Every copy
> keeps the behaviour. Each removes 64% to 82% of the excess that leaving the update in Δ shows on the check. The
> copies agree on every other number; they differ only in what the adversary finds on one block of 128 Pile rows.
> One new order read 17% below the first, against at most 11% for finished copies at a second order. Each reading
> had one adversary start, so how much of that is batch order is untested. On the second task the step-10 copy passed
> at 1.38, 1.4% inside the line, at one order. The adjoint split of the same emoticon update passed at 1.06 with
> P(target) 0.93, and loses the behaviour later in the fine-tune (E22).
>
> **The options.**
> - (a) Accept "at the line" for the step-10 update on this model, and move on. No cost, but the short-links run then
>   reads its early links at one order.
> - (b) Read the three copies at adversary seeds 1 and 2 as well, reporting each start and keeping the verdict at
>   seed 0. This tells whether the spread is the adversary's or the copy's, and so whether three adversary starts on
>   one copy could replace three batch orders. About $2; a departure from the one-start decision, though not a search.
> - (c) Train the step-10 copy longer, 3,000 steps at three orders. About $2.5. Against it: the copy already matches
>   its teacher closely, and longer training did not lower the check on E37's broad update.
> - (d) Start teacher training from the scaled adjoint split, which passes at step 10, for early teachers. Small new
>   code, a review and about $1.5. Against it: from that start, E37's broad copy went from 1.05 to 1.76 within 1,000
>   steps.
> - (e) Read every early-step copy at three batch orders and score the median. About 4 more minutes per cell here. It
>   triples the short-links chain's run time, about $1.5 to roughly $4, and costs much more on Qwen.
>
> **Recommendation.** (a) with (e). The three orders agree that the copy keeps the behaviour and removes most of the
> excess; they disagree only on which side of the pass line it lands. A median of three is a robust reading whichever
> noise causes that. If a cheaper standard matters, for Qwen especially, (b) first tells whether three adversary starts
> would do instead. (c) and (d) would test fixes for a shortfall whose cause is untested.
>
> **Why it matters.** Small updates are the case `GOALS.md` cares about. They are also where continuous
> interpretability would apply teacher training step after step. A single early reading near the line cannot be read
> on its own.

The Blocking flag is fair: the decision changes the cost of the approved short-links run by about three times.

## 5. Smaller points

- **Paths moved during this check.** At 13:59 `experiments/teacher_training/results/e41a1_dl/` and `e41_dl/` were
  emptied. The logs are now at `results/large/run_e41_amend1.log` and `results/large/run_e41.log`, with the sha256
  values the analysis lists. The section's header and provenance, the notebook's last Amendment 1 cell, and
  `plot_e41_amend1.py`'s comment on line 28 still name the `_dl` paths.
- **Figure provenance.** The section says the figure comes "from the two results files". `plot_e41_amend1.py` line 30
  also hard-codes order 0's block values for both teachers, from E41's log lines 149-150 and 164-165. They are
  correct, but the provenance line should say so.
- **Figure.** The title's "batch order moves it 21%" carries the attribution of claim 1; "spans 21% across batch
  orders" fits. The grey band is W₀'s spread for all 256 rows. The triangles are per-block, and W₀'s second-block
  range (up to 1.15) is not drawn, so the second-block triangles are read against the wrong reference.
- **Notebook.** It repeats claims 1, 2, 3, 9 and 10 and § 1 items 1, 2 and 4 with the same wording. It needs the same
  corrections. Its "by more than the unchanged model's own spread across adversary starts" should say "just more".

## 6. Wording I would allow above the cards

> **Early updates (E41 and its Amendment 1; one fine-tune seed, one adversary start per reading, 67M).** Teacher
> training keeps the behaviour of a fine-tune stopped at step 10, the one step where leaving the update in Δ fails the
> adversarial check. The adversarial reading there sits at the pass line. On emoticon, three copies trained at
> different batch orders read 1.55, 1.28 and 1.51: one pass and two inconclusive, none failing. Each removed 64% to 82%
> of the excess that doing nothing shows. The copies agree on every other number. They differ by up to a fifth in what
> the adversary finds on one block of 128 Pile rows, more than any finished copy has moved between orders. With one
> adversary start each, whether that is batch order or the adversary is untested. On the second task the step-10 copy
> passed by 1.4%, at one order.

Do not carry into the log: "batch order moved this copy's reading by a sixth", "more sensitive to batch order", "removes
most of the damage", "two-thirds or more", "least certain on the earliest, smallest updates", "works on fine-tunes run
to the end", "one order alone misread it", or "like for like" for the spread comparison.

## Not checked

- `path_amend1.py` beyond `near_line`, `score`, `LINES` and the start-KL line; `amend4.train_student_seeded` itself
  (G2's exact reproduction stands in for it).
- E22's 1.06 and P 0.93 against E22's own files (taken from the E41 check, which recomputed them).
- Pod cost and pod time (not in the repo; the commit message says about $1.1).
- `check.py` on the analysis section (the author's step).
