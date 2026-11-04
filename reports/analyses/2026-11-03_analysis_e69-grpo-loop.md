# Does RL on Qwen3-1.7B gain more than the same loop with shuffled rewards?

*E69, 2026-11-03. Plan `reports/plans/2026-11-03_plan_e69-grpo-loop.md`, notebook `notebooks/E69_qwen_rl.ipynb`, numbers
`experiments/qwen_rl/results/lr1e-6w10/{analysis,rollouts_summary,cosines}.json`, figures `experiments/qwen_rl/figures/`. Wording of the claims revised the same day after `reports/reviews/2026-11-03_review_e69-independent-check.md`; no filed prediction changed. Amendment 1, 2026-11-04: plan `reports/plans/2026-11-03_plan_e69-amend1.md`, rules `results/predictions_amend1.json`, scores `results/amend1_score.json`, unfiled readings `results/amend1_reading.json`. An independent check ran on 2026-11-04 (`~/commute-data/scratch/qwen-rl-lead/e69-a1-check/`). It reproduced every number and label, matched the per-problem files' checksums, and found no leakage, no mix-up and no bug in the headline. It changed the wording on pools and on the saved-step graders. It did not recompute the δW hashes, which the fetch had checked against the run's checksums (job 51098).*

We trained Qwen3-1.7B on maths problems by reinforcement learning, by the same loop with rewards shuffled, and by supervised training on its own correct answers, then repeated the first two with a second seed. Under the exact-format grader, reinforcement learning beat the shuffled loop on the same test problems by about three and six points, and with format set aside the second seed stays near six while the first ranges from one to five and a half with how long answers are treated. The updates are ready to go forward to the decomposition, an independent check having reproduced these numbers, with the shuffled loop read as a large change in behaviour without signal and not as a do-nothing.

## The answer

Under the strict grader, reinforcement learning beat the shuffled-reward null on the same test problems on both seeds, by about three and six points. With format set aside, the second seed's lead stays about six however long answers are treated. The first seed's runs from about one point within noise on answers all three states finish, through two and a half on whole answers, to five and a half if an answer past the window counts as wrong, as the reward counts it. That first-seed null ran long more often, and was right more often on answers that finished; we cannot explain why. Our filed lean towards no clear lead reversed on the second seed. That seed's null just cleared its strict band and reads zero with format set aside. RL states the right number six to seven points more often than the unchanged model on GSM8K test, with format set aside, but one shuffled run came close on answers that finish and our graders do not read the working, so we do not claim the reward taught better maths. Two seeds cannot give the lead's size. An independent check reproduced these numbers on 2026-11-04.

**The numbers.** All are RL minus the null on GSM8K test, with the standard error over problems at fixed weights in brackets. The filed pool is answers all three states finish. There the first seed reads +3.21 (0.99) strict and +1.25 (0.90) format-relaxed, on 1,123 problems, and the second seed +6.42 (1.07) under both, on 1,200 problems. The first seed's format-relaxed lead is 1.4 standard errors from zero, and the filed rule for the second seed would label it "ambiguous". The other two pools are unfiled and were read after the fact.

| format-relaxed lead, GSM8K test | first seed | second seed | second minus first, same problems |
|---|---|---|---|
| answers all three states finish: filed | +1.25 (0.90) | +6.42 (1.07) | 5.17 on each seed's own pool; +4.51 (1.35) on the 1,108 all five states finish |
| whole answers: read after the fact | +2.50 (0.97) | +6.44 (1.08) | +3.94 (1.39) |
| an answer past the window counts as wrong: read after the fact | +5.53 (1.02) | +6.52 (1.09) | +0.99 (1.44) |

Under the strict grader, the second seed minus the first on the 1,108 shared problems is +2.53 (1.41). The two graders agree on the second seed because they differ on 0 of 1,319 answers per arm, and on 0 of 1,197 on GSM1K. The first seed's null writes "The answer is N" in 1,124 of its 1,154 answers that fit. So "the lead may be largely answer format" holds for the first seed on the filed pool, and not for the second seed. That first-seed null fitted fewer answers than the second seed's, 1,154 against 1,277, and was right on more of those that fit, 89.0% against 79.4%. This is an accounting of the gap, not a cause.

**The claim we can make, in plain words.** Under the strict grader, RL's update raised accuracy on unseen problems more than a shuffled-reward update did. That holds on both seeds, on both test sets, and under every way of treating long answers. With format set aside the same sign holds in every cell of the table below, and on the second seed every cell stands clear of noise. On the first seed only some do. RL's own gain is close between seeds on every pool. Format-relaxed, on problems kept at that arm and the unchanged model, RL reads +7.23 and +6.58 and the null +5.80 and +0.16. If an answer past the window counts as wrong, RL reads +7.28 and +7.43 and the null +1.74 and +0.91. So the null's reading is what moves, and on the last pool it barely moves. That does not say why.

On the first seed, the lead is largest when an answer past the window counts as wrong, as it does for the reward, and most of it there is the null's answers not finishing. On sampled answers read that way, RL climbs to five points above the unchanged model, with a standard error of 1.1 to 1.2, while the null stays within two points of it. RL's sampled lead reads +5.16 (0.69) in the last five steps and +2.95 (0.36) averaged over the last fifty, with standard errors over steps. The second seed, now decoded, reads +4.53 (1.87) and +4.52 (0.42).

**The claim we cannot make yet** is that RL does better maths than the null, that RL learned maths at all, or what size the lead is. On the filed pool two seeds give a format-relaxed lead of +1.25 and +6.42, so neither number stands for the other, and their average would hide what is unexplained. RL's gains overstate what the reward added, since format is part of them and the first seed's null gained nearly as much format-relaxed. The graders are also coarse: the format-relaxed one reads the stated number and the lenient one the last number, and neither checks the working. On the first seed only, keeping a problem because the null finished it may favour the null. Kept at the unchanged model alone, whole answers there read +2.15 and +1.27.

| RL minus null, same problems, points (SE) | drop rule | whole answers | capped answers |
|---|---|---|---|
| **First seed.** GSM8K test, strict grader | +3.2 (1.0) | +4.6 (1.0) | +7.4 (1.1) |
| GSM1K, strict grader | +3.1 (1.2) | +4.8 (1.2) | +8.4 (1.2) |
| GSM8K test, format-relaxed grader | +1.25 (0.90) | +2.50 (0.97) | +5.53 (1.02) |
| GSM1K, format-relaxed grader | +0.81 (1.06) | +3.09 (1.15) | +6.43 (1.15) |
| GSM8K test, lenient grader | +1.2 (0.9) | +2.1 (1.0) | +4.3 (1.0) |
| GSM1K, lenient grader | +0.7 (1.1) | +2.4 (1.1) | +6.3 (1.2) |
| GSM8K test, lenient, kept at the unchanged model only | | +2.15 (0.96) | |
| GSM1K, lenient, kept at the unchanged model only | | +1.27 (1.19) | |
| **Second seed.** GSM8K test, strict grader | +6.42 (1.07) | +6.44 (1.08) | +6.52 (1.09) |
| GSM1K, strict grader | +4.67 (1.22) | +6.43 (1.21) | +4.18 (1.16) |
| GSM8K test, format-relaxed grader | +6.42 (1.07) | +6.44 (1.08) | +6.52 (1.09) |
| GSM1K, format-relaxed grader | +4.67 (1.22) | +6.43 (1.21) | +4.18 (1.16) |
| GSM8K test, lenient grader | +6.58 (1.08) | +6.67 (1.08) | +6.75 (1.09) |
| GSM1K, lenient grader | +4.67 (1.22) | +6.68 (1.21) | +4.43 (1.19) |

The drop-rule column keeps problems that fit at the unchanged model, RL and the null alike: 1,123 and 868 on the first seed, 1,200 and 964 on the second. The rows kept at the unchanged model only have 1,255 and 1,023 problems and no selection on the null's outcome, and they come from `reports/reviews/2026-11-03_review_e69-independent-check.md`. The format-relaxed grader accepts "the answer is N" in either case wherever it stands, and the `\boxed` and bold wrappers directly after it. Only the GSM8K drop-rule cells of the strict and format-relaxed rows are scored by a filed rule. The other second-seed cells, and the whole-answer and capped cells of the format-relaxed rows, are unfiled, read after the fact, from `results/amend1_reading.json`. On GSM1K the format-relaxed capped lead runs the other way between seeds, second minus first −2.26 (1.59), against +3.34 (1.57) on whole answers. These standard errors count problems at fixed weights, so they carry no spread between seeds. The two test sets are two samples of problems read from the same weights, so their agreement within a seed does not replicate the arm difference.

## What we were trying to test

The project needs real post-training updates to carry through a decomposition, and no reinforcement-learning update of this model existed. We trained the unchanged model, with thinking off, for one hundred steps on GSM8K train, three ways, on the same prompts at each step.

- **RL.** 64 new problems per step, 8 samples each at temperature 1. Reward is 1 for a strictly right answer that ends inside the 512-token window, else 0. Weight is reward minus the problem's mean reward.
- **The shuffled-reward null.** The same, sampling from its own changing weights, with rewards shuffled among each problem's samples. It rules out that any update of this size moves accuracy by itself.
- **Own-samples SFT.** Weight 1 on the unchanged model's correct samples, drawn once and never redrawn, on the same prompts.

All use AdamW at a rate of one in a million, reached over ten warm-up steps. Accuracy is the greedy answer. The strict grader takes the last "The answer is N" anywhere in the response; the lenient grader takes the last number.

Long answers need a rule, and `qwen_eval.summary` defines three. The **drop rule**, the filed one, keeps problems whose answers fit the window at both states. **Whole answers** grades every problem on its full response, up to 512 new tokens. **Capped answers** grades every problem on the text the window keeps.

**The amendment.** One seed per arm could not say whether the lead or the null's gain would come back, so three things were added.

- **A second seed.** RL and the shuffled-reward null again, from the unchanged model, on the same recipe and test problems, with a different draw seed. It rules out reading one walk of each arm as the arm.
- **Saved steps.** Greedy answers at steps 10, 20 and 50 of both arms on both seeds, on GSM8K test only. They rule out reading step 100 as the end of a steady climb.
- **Forced openings, first seed only.** The unchanged model with its answers prefilled "We are" or "We need", and the null at step 100 prefilled with the unchanged model's own first two words. They test whether the opening words carry the null's gain.

The **format-relaxed grader** was added to read the lead without the answer line's exact form. It is blind to the maths beyond the stated number. It also misses some forms, such as "the answer is:" followed by a display `\boxed{230}`, and "\text{The answer is } \boxed{50}". On the first seed's null at step 50, 39 answers are lenient-right and format-relaxed-wrong. At step 100 the two differ on at most one fitting answer per arm, so no filed number moves, and `relaxed.py` is unchanged because filed rules read it.

## What we expected, and why

We expected RL to gain three to eight points on test and about the same on GSM1K. We expected the null to stay within a point and a half, because its expected gradient is zero. One update per batch also removes the clipping that the scoping report, citing Spurious Rewards, blames for gains under random rewards. We expected the right-over-wrong signal to grow in RL only. Whether SFT moves the policy more than RL was left open.

For the second seed we filed these before reading it. RL's lead would replicate under the strict grader, since the first seed read +3.21. Under the format-relaxed grader we leaned towards an ambiguous reading or no lead, since the first seed read +1.25. A lead replicates at two points and two standard errors, and under one point is no lead. The null's gain counts as systematic at 1.5 points or more with the first seed's sign. We expected RL's lenient gain at step 10 to be within a point of zero, and forcing "We are" on the unchanged model to add nothing.

## Question 1: paired gains from the unchanged model

Figures: `figures/gains.png`, `figures/grader.png`. Points, with paired SE over problems. The filed numbers are the drop-rule column of the strict rows.

| set | grader | arm | drop rule | n | whole answers | capped answers |
|---|---|---|---|---|---|---|
| GSM8K test | strict | RL | +9.04 (1.13) | 1217 | +9.40 (1.15) | +9.02 (1.15) |
| | | null | +5.63 (1.19) | 1137 | +4.85 (1.23) | +1.67 (1.20) |
| | | SFT | +2.58 (0.74) | 1240 | +3.18 (0.78) | +2.73 (0.73) |
| GSM1K | strict | RL | +7.08 (1.19) | 975 | +7.77 (1.18) | +6.43 (1.16) |
| | | null | +3.50 (1.31) | 886 | +3.01 (1.23) | -1.92 (1.24) |
| | | SFT | +1.19 (0.90) | 1009 | +2.67 (0.92) | +1.67 (0.84) |
| GSM8K test | lenient | RL | +6.74 (1.07) | 1217 | +6.29 (1.10) | +6.82 (1.11) |
| | | null | +5.36 (1.07) | 1137 | +4.17 (1.13) | +2.58 (1.13) |
| | | SFT | +0.56 (0.61) | 1240 | +0.53 (0.64) | +0.53 (0.65) |
| GSM1K | lenient | RL | +5.64 (1.13) | 975 | +6.02 (1.13) | +5.68 (1.13) |
| | | null | +4.63 (1.16) | 886 | +3.59 (1.14) | -0.58 (1.17) |
| | | SFT | +0.00 (0.83) | 1009 | +0.84 (0.85) | +0.42 (0.82) |

**Reading it.** RL gains under every rule, and the rule barely changes its number. The null ranges from a clear gain to nothing as long answers count against it. SFT's strict gain is almost all format, since its lenient gain is near zero on both sets. On GSM8K test, no RL answer that fits the window is right yet missing the answer line, while the unchanged model has 29. This table is the first seed's. Question 2 gives the second seed's gains.

## Question 2: does the lead come back on a second seed?

Figure: `figures/leads.png`. Points, with SE over problems at fixed weights. Filed pool first.

| GSM8K test, on answers all three states finish | first seed, 1,123 problems | second seed, 1,200 problems |
|---|---|---|
| RL minus null, strict | +3.21 (0.99) | +6.42 (1.07) |
| RL minus null, format-relaxed | +1.25 (0.90) | +6.42 (1.07) |
| RL minus null, lenient | +1.16 (0.89) | +6.58 (1.08) |

| gain from the unchanged model, on answers both states finish | first seed: RL | second seed: RL | first seed: null | second seed: null |
|---|---|---|---|---|
| GSM8K test, strict: the filed reading | +9.04 (1.13) | +8.31 (1.11) | +5.63 (1.19) | +1.62 (1.11) |
| GSM8K test, format-relaxed | +7.23 (1.08) | +6.58 (1.08) | +5.80 (1.09) | +0.16 (1.06) |
| GSM8K test, lenient | +6.74 (1.07) | +6.17 (1.06) | +5.36 (1.07) | −0.41 (1.05) |
| GSM1K, strict | +7.08 (1.19) | +5.94 (1.22) | +3.50 (1.31) | +1.10 (1.33) |
| GSM1K, format-relaxed | +5.85 (1.14) | +4.82 (1.17) | +4.74 (1.17) | +0.00 (1.29) |
| GSM8K test, strict, whole answers | +9.40 (1.15) | +8.64 (1.13) | +4.85 (1.23) | +2.20 (1.14) |
| GSM8K test, format-relaxed, whole answers | +6.75 (1.11) | +5.99 (1.08) | +4.25 (1.14) | −0.45 (1.07) |
| GSM8K test, strict, an answer past the window counts as wrong | +9.02 (1.15) | +9.17 (1.14) | +1.67 (1.20) | +2.65 (1.11) |
| GSM8K test, format-relaxed, an answer past the window counts as wrong | +7.28 (1.11) | +7.43 (1.12) | +1.74 (1.14) | +0.91 (1.07) |
| GSM1K, format-relaxed, an answer past the window counts as wrong | +5.43 (1.13) | +6.18 (1.18) | −1.00 (1.18) | +2.01 (1.23) |

The last three rows, the capped ones, and the whole-answer rows are unfiled and read after the fact.

**Reading it.** Yes, and more than we filed. A lean we filed reversed: under the format-relaxed grader we expected an ambiguous reading or no lead, and the second seed reads +6.42, past the rule's two points and two standard errors. The rule reads the second seed alone. The first seed's +1.25 is 1.4 standard errors from zero and would read "ambiguous" under it. The strict lead also replicates, as filed.

**Why the two graders agree on the second seed.** The strict and format-relaxed grades differ on 0 of 1,319 answers per arm, and on 0 of 1,197 on GSM1K, because every answer that fits the window holds "The answer is N". The counts are 1,255 of 1,255 for RL and 1,277 of 1,277 for the null, and on GSM1K 1,044 of 1,044 and 1,081 of 1,081. On the first seed, RL's answers do too, with 1,249 of 1,249. Its null's do not, with 1,124 of 1,154, and 23 of the null's answers there change grade between the two graders. That is what separates +3.21 from +1.25. So the earlier reading "the lead may be largely answer format" is a reading of the first seed on the filed pool, and the second seed does not share it.

**The spread between seeds depends on the pool, and is unexplained where it stands.** The five-point gap belongs to the filed pool. Format-relaxed, it is 5.17 points there, +3.94 (1.39) on whole answers and +0.99 (1.44) if an answer past the window counts as wrong, where the two seeds cannot be told apart. Strict, on the problems all five states finish, it is +2.53 (1.41). In accounting terms, the first seed's null fitted fewer answers, 1,154 against 1,277, and was right on more of those that fit, 89.0% against 79.4%. No cause is named for either. RL's own gains differ by about a point or less on these pools, which locates the difference in the null's reading and does not explain it.

**The null gained again, only just.** On GSM8K test its filed gain reads +1.62 (1.11) against a band of 1.5, so the filed label is "systematic" by 0.12 points, and the rule has no standard error in it. The first seed read +5.63. GSM1K reads +1.10 (1.33). With format set aside the second seed's gain is +0.16 (1.06) format-relaxed and −0.41 (1.05) lenient, where the first seed's read +5.80 and +5.36.

**The first seed's drift in form and length did not repeat.** The first seed's null dropped the exact answer line in 30 of its fitting answers, grew to 255 tokens on average, and fitted the window on 87.5% of problems. The second seed's null dropped it in none, reads 198 tokens against the unchanged model's 199, and fits on 96.8%. Its sampled reward does not collapse either: the mean over steps 76-95 is 0.83, against 0.62 on the first seed, and its lowest is 0.70 at step 13. None of this is offered as the cause of either null's gain.

| other readings at step 100 | first seed: RL | second seed: RL | first seed: null | second seed: null |
|---|---|---|---|---|
| size of the update | 0.351 | 0.347 | 0.370 | 0.335 |
| KL from the unchanged model | 0.194 | 0.196 | 0.321 | 0.137 |
| right-over-wrong signal ΔR | 8.69 | 7.32 | 5.10 | 4.38 |

## Question 3: what the null's gain consists of

Figures: `figures/accounting.png`, `figures/openings.png`, `figures/lengths.png`, all from the first seed, and `figures/saved_steps.png` for both seeds. The accounting, the openings table and the sampled tables below are the first seed's unless a row says otherwise.

**The accounting.** Net problems right, arm minus unchanged model, as wrong-to-right and right-to-wrong counts.

| GSM8K test, 1,319 | n | whole answers | capped answers |
|---|---|---|---|
| null: fits at both | 1137 | +125 −61 = **+64** | +125 −61 = **+64** |
| null: fits before only, ran long | 118 | +19 −30 = −11 | +0 −53 = −53 |
| null: fits after only, came back | 17 | +12 −1 = +11 | +12 −0 = +12 |
| null: fits at neither | 47 | +9 −9 = 0 | +0 −1 = −1 |
| null: total | 1319 | **+64 = +4.85 points** | **+22 = +1.67 points** |
| RL: fits at both | 1217 | +154 −44 = +110 | +154 −44 = +110 |
| RL: ran long | 38 | +8 −9 = −1 | +0 −16 = −16 |
| RL: came back | 32 | +18 −2 = +16 | +25 −0 = +25 |
| RL: fits at neither | 32 | +2 −3 = −1 | 0 |
| RL: total | 1319 | **+124** | **+119** |

| GSM1K, 1,197 | n | whole answers | capped answers |
|---|---|---|---|
| null: fits at both | 886 | +83 −52 = **+31** | +83 −52 = **+31** |
| null: ran long | 137 | +19 −25 = −6 | +0 −68 = −68 |
| null: came back | 20 | +4 −1 = +3 | +15 −0 = +15 |
| null: fits at neither | 154 | +20 −12 = +8 | +1 −2 = −1 |
| null: total | 1197 | **+36 = +3.01 points** | **−23 = −1.92 points** |

**What this establishes.** The filed 5.63 is answers that changed on problems kept at both states: 64 net problems out of 1,137. Which problems are dropped matters in a specific way. Count the 118 that left the window by their whole answers, and the gain over all problems that fit at the start is 4.22. Count them by their capped text, and it is 0.88.

So the capped number differs for one reason. On GSM8K test, 53 problems the unchanged model got right now run past the window, and 23 of them are right if allowed to finish. RL lost 16 this way. On GSM1K the null lost 68, which takes it below zero. Unpaired, the drop rule would read 8.83, so pairing had already removed 3.2 points of selection.

**How lengths moved.**

| greedy answers | mean tokens, GSM8K test | mean tokens, GSM1K | fits the window, GSM8K test | hits the token cap, GSM8K test |
|---|---|---|---|---|
| unchanged model | 199 | 227 | 95.1% | 1.3% |
| RL | 221 | 243 | 94.7% | 2.1% |
| null | 255 | 284 | 87.5% | 6.1% |
| SFT | 196 | 224 | 95.8% | 1.0% |
| RL, second seed | 186 | 215 | 95.1% | 2.5% |
| null, second seed | 198 | 222 | 96.8% | 0.7% |

**What changed in the answers.** On the first seed, all of the null's greedy answers on both sets open "We are" or "We need". At the unchanged model 19 do on GSM8K test, and none of the null's answers is identical to the unchanged model's. RL's answers moved another way: "Step 1:" went from 425 to zero and "Let's" took over. SFT stayed close, with "Step 1:" rising to 540 and 420 answers identical word for word.

On the second seed the words differ. All 1,319 of the null's greedy answers on GSM8K test open "Step 1:", and none opens "We". RL's answers have no common opening: the most frequent, "There are", opens 30.

| greedy opening, GSM8K test | unchanged model: n, right | RL: n, right | null: n, right | SFT: n, right |
|---|---|---|---|---|
| Let's break | 498, 0.681 | 991, 0.826 | | 452, 0.717 |
| Step 1: | 425, 0.814 | | | 540, 0.828 |
| Let's solve | 183, 0.776 | 273, 0.938 | | 181, 0.801 |
| We are | 19, 0.684 | | 1023, 0.828 | 11, 0.818 |
| We need | | | 296, 0.733 | |

The same flip happens in sampled answers, and early. "We" openings go from 3% to 22% by steps 11-20, 83% by steps 21-50, and all by the last steps. RL's "Let" openings go from 31% to 93% over the same steps.

**Explanations tried, and how they fared.**

1. *Selection by the drop rule.* Broken as the source of whole-answer accuracy, which is 4.85 with nothing dropped. With an answer past the window counted as wrong, the gain over all problems that fit at the start falls to 0.88, because the dropped answers count there.
2. *Longer answers are more often right.* Not supported problem by problem. Kept problems going wrong to right lengthened by 48 tokens, and those going right to wrong by 45. The correlation is 0.018 on GSM8K test and −0.027 on GSM1K. This test is blind to a uniform change of style.
3. *Format.* Broken for the null. Its lenient gain under the drop rule is 5.36 against 5.63 strict.
4. *Self-checking.* Phrases such as "double-check" and "wait" rose from 0.7 to 12.3 percent of answers, by a crude count. Those answers gain 7.6 points with se 4.4, and the others gain 5.5 with se 1.2. So they do not carry it.
5. *The opening flipped, and the new opening is better.* This was our favoured hypothesis, and its most direct test counts against it. In the unchanged model's own 51,200 samples, "We" openings can be compared with "Let" openings on the same problem, across 644 problems. Ignoring format and counting only answers inside the window, "We" is 2.5 points worse with se 1.2. It is also 9.5 points worse on reward with se 1.5, and 20 tokens longer. "Step" against "Let" on the same problem reads 0.0 with se 0.7, so the large differences between openings in the table above are which problems each opening goes with. The test reads samples of the unchanged model, not greedy answers of the null. The forced openings below test it directly, and their two filed rules read opposite ways.
6. *Some wrong answers are easy to flip by any change.* Partly supported. Of the 199 problems the unchanged model gets wrong and that fit at all four states on GSM8K test, the null gets 117 right, RL 129, and 97 both. If independent, 76 would be right in both. Only 48 stay wrong in all three arms. Whole-answer gains are largest on the third of problems with the longest answers: 12.2 for the null, 15.2 for RL and 7.9 for SFT. This cannot alone give a net gain, since the null's flows are 125 against 61.
7. *A bad file or a changed evaluation.* Broken. The unchanged model's files match the run's checksums, and the same code graded every state. SFT, nearest to the unchanged model, gains near zero under the lenient grader.

**The sampled answers settle part of it.** Figures: `figures/reward.png`, `figures/collapse.png`. The rollouts arrived before this report was done, with checksums matching, and were decoded for $0. The zero line is the unchanged model on the same prompts and seeds, read from SFT's rollouts.

| steps | reward: RL − unchanged | reward: null − unchanged | format ignored: RL − unchanged | format ignored: null − unchanged | format ignored: RL − null |
|---|---|---|---|---|---|
| 1-10 | +2.2 (0.5) | +1.5 (0.4) | +0.1 (0.5) | +0.2 (0.4) | −0.2 (0.7) |
| 11-20 | +2.8 (0.9) | +1.7 (0.3) | +0.3 (0.9) | −0.2 (0.4) | +0.5 (0.8) |
| 21-50 | +5.1 (0.4) | −2.0 (0.4) | +2.5 (0.3) | −0.6 (0.4) | +3.0 (0.5) |
| 51-75 | +5.9 (0.5) | −0.3 (0.7) | +3.5 (0.4) | +1.6 (0.4) | +1.8 (0.4) |
| 76-95 | +6.8 (0.6) | −19.2 (1.2) | +4.1 (0.6) | +0.3 (0.4) | +3.8 (0.6) |
| 96-100 | +7.3 (1.1) | −3.8 (1.4) | +5.0 (1.1) | −0.2 (1.6) | +5.2 (0.7) |

"Format ignored" is the share of samples that are lenient-right and end inside the window. Standard errors are over steps of this one run.

| steps | has the answer line, of those inside: unchanged, RL, null | lenient-right, of those inside: unchanged, RL, null | inside the window: unchanged, RL, null |
|---|---|---|---|
| 1-10 | 0.964, 0.992, 0.980 | 0.849, 0.834, 0.850 | 0.956, 0.975, 0.959 |
| 21-50 | 0.964, 0.999, 0.949 | 0.881, 0.886, 0.900 | 0.955, 0.978, 0.929 |
| 76-95 | 0.966, 1.000, 0.736 | 0.871, 0.898, 0.903 | 0.964, 0.980, 0.933 |
| 96-100 | 0.975, 1.000, 0.923 | 0.847, 0.896, 0.896 | 0.955, 0.959, 0.902 |

- **The null's collapse is the answer line, not the maths.** Between steps 76 and 95 its reward fell by 19 points, to 0.53 at step 88. Over those steps the share of finished samples with the answer line fell from 0.95 to 0.74, while the lenient-right share stayed at 0.90. That line recovered to 0.92 by the last five steps.
- **With an answer past the window counted as wrong, the null gains little or nothing.** Format ignored, its samples read −0.2 with se 1.6 over the last five steps, and RL's +5.0 with se 1.1. Its capped greedy accuracy agrees: +2.6 and −0.6 on the lenient grader.
- **Answers that finish are as often right in the null as in RL,** 0.896 in both at the end. But fewer of the null's finish, and the finishers are selected. So this does not show the null's maths improved.
- **RL's earliest reward is format.** It gains about two points by step 20 and none with format ignored; the answer line reaches 1.00 by then.
- **A small unexplained detail.** The null also gained about 1.5 points of reward over steps 1-20, and all of it is the answer line, 0.98 to 0.99 against 0.96.

**The second seed's samples,** decoded for $0, `results/lr1e-6w10_s70/rollouts_summary.json`. These are read as they came, with no rule filed and no cause named. There is no SFT arm on this seed. The unchanged model here is the first seed's SFT rollouts, on the same prompts in the same order at all 100 steps but with other draws, so its zero line is not paired on draws as the first seed's is. Standard errors are over steps. RL minus null is on this seed's own rollouts.

| steps | reward: RL − unchanged | reward: null − unchanged | format ignored: RL − unchanged | format ignored: null − unchanged | format ignored: RL − null, second seed | same, first seed |
|---|---|---|---|---|---|---|
| 1-10 | +2.9 (0.8) | −0.1 (0.5) | +0.9 (0.5) | +0.3 (0.2) | +0.6 (0.5) | −0.2 (0.7) |
| 11-20 | +1.5 (0.6) | −2.1 (1.3) | −1.1 (0.6) | −0.7 (0.6) | −0.4 (1.0) | +0.5 (0.8) |
| 21-50 | +4.1 (0.4) | +0.4 (0.3) | +1.5 (0.4) | −0.3 (0.3) | +1.8 (0.4) | +3.0 (0.5) |
| 51-75 | +5.9 (0.5) | +1.5 (0.6) | +3.5 (0.4) | −0.6 (0.6) | +4.1 (0.7) | +1.8 (0.4) |
| 76-95 | +6.8 (0.6) | +1.7 (0.6) | +4.1 (0.6) | −1.0 (0.6) | +5.1 (0.5) | +3.8 (0.6) |
| 96-100 | +6.8 (0.4) | +2.2 (2.1) | +4.5 (0.6) | −0.1 (1.7) | +4.5 (1.9) | +5.2 (0.7) |

| steps, second seed | has the answer line, of those inside: RL, null | lenient-right, of those inside: RL, null | inside the window: RL, null | most common first word of the null's samples |
|---|---|---|---|---|
| 1-10 | 0.988, 0.959 | 0.853, 0.856 | 0.963, 0.953 | Let, 61% |
| 11-20 | 1.000, 0.948 | 0.830, 0.883 | 0.978, 0.923 | Let, 79% |
| 21-50 | 1.000, 0.978 | 0.868, 0.893 | 0.987, 0.939 | Let, 61% |
| 51-75 | 0.999, 0.994 | 0.886, 0.844 | 0.975, 0.975 | Step, 72% |
| 76-95 | 1.000, 1.000 | 0.895, 0.840 | 0.983, 0.988 | Step, 100% |
| 96-100 | 1.000, 1.000 | 0.883, 0.829 | 0.967, 0.976 | Step, 100% |

- **RL's sampled lead comes back.** Format ignored and an answer past the window counted as wrong, RL ends about five points above the second seed's null, as it does on the first seed. The last-fifty mean is +4.52 (0.42), against +2.95 (0.36).
- **RL's own climb repeats.** It reads +4.5 over the last five steps against the unchanged model, where the first seed read +5.0.
- **The null stays flat again.** Its format-ignored reading ends at −0.1 with se 1.7 against the first seed's −0.2 with se 1.6.
- **The collapse does not repeat.** Its answer line never falls below 0.948, and reaches 1.000 by steps 76-95.
- **One thing differs.** Answers that finish are less often right in this null than in RL at the end, 0.829 against 0.883. On the first seed they matched, at 0.896. Finishers are selected, so this does not measure maths.

**How firmly it stands.**

- The accounting is arithmetic on the per-problem files, and it stands.
- The collapse being format stands, from the decoded text of one run.
- The null's real whole-answer gain, 4.85 with se 1.23, stands as a reading of the first seed at step 100. The second seed's is 2.20 with se 1.14.
- Its cause is unexplained: we can rule out format, selection, and length problem by problem, on this seed. The opening stays open, with its two forced tests reading opposite ways.

**When the gain arrives: the saved steps.** Figure: `figures/saved_steps.png`. Gain from the unchanged model in points, whole answers, GSM8K test, all 1,319 problems. The unchanged model is the zero line: 75.8% right under the strict grader and 78.5% format-relaxed.

| arm, seed, grader | step 10 | step 20 | step 50 | step 100 |
|---|---|---|---|---|
| RL, first seed, strict | +1.06 (1.12) | +4.85 (1.13) | +6.37 (1.11) | +9.40 (1.15) |
| RL, second seed, strict | +3.79 (1.09) | +2.73 (1.16) | +6.97 (1.15) | +8.64 (1.13) |
| null, first seed, strict | +4.17 (0.96) | +5.99 (1.15) | +1.67 (1.25) | +4.85 (1.23) |
| null, second seed, strict | +0.45 (1.05) | +4.32 (1.10) | +0.30 (1.10) | +2.20 (1.14) |
| RL, first seed, format-relaxed | −1.59 (1.06) | +2.20 (1.08) | +3.79 (1.05) | +6.75 (1.11) |
| RL, second seed, format-relaxed | +1.14 (1.04) | +0.08 (1.12) | +4.32 (1.11) | +5.99 (1.08) |
| null, first seed, format-relaxed | +1.52 (0.88) | +3.56 (1.10) | −0.99 (1.22) | +4.25 (1.14) |
| null, second seed, format-relaxed | +1.21 (0.97) | +1.67 (1.06) | −2.20 (1.05) | −0.45 (1.07) |
| RL, first seed, lenient | −1.97 (1.04) | +1.82 (1.07) | +3.41 (1.03) | +6.29 (1.10) |
| RL, second seed, lenient | +0.83 (1.03) | −0.38 (1.10) | +4.02 (1.09) | +5.69 (1.07) |
| null, first seed, lenient | +1.21 (0.86) | +3.71 (1.07) | +1.52 (1.12) | +4.17 (1.13) |
| null, second seed, lenient | +1.44 (0.97) | +1.90 (1.03) | −2.50 (1.03) | −0.99 (1.05) |

Steps 10, 20 and 50 were read on an L40 and come from `amend1_score.json`. Step 100 was read on the H100 and comes from `amend1_reading.json`, computed by the same paired formula. The unchanged model's grade and fit match between the two cards on all 1,319 problems. The first seed's null rebuilt at step 100 on the L40 reads +4.78 (1.23) strict, one answer apart.

**The format-relaxed rows still hold format at saved steps,** which is why the lenient rows sit beside them here and in the figure. The grader misses forms such as "the answer is:" followed by a display `\boxed{230}`. On the first seed's null at step 50 that reads −0.99 format-relaxed against +1.52 lenient, with 39 answers lenient-right and format-relaxed-wrong. So use the lenient row to read format-free movement at steps 10 to 50.

**Reading it.** No single step here is a trend. Under the strict grader the null moves up and down by about four points between saved steps on both seeds, so its step-100 reading is one point of a walk. RL rises across the steps on both seeds, with one dip on the second. Under the lenient grader the first seed's null moves by about two and a half points and the second seed's by up to four and a half. The whole-answer gaps at step 100, +4.55 and +2.50 on the first seed and +6.44 on the second, are the middle row of the headline's table.

The filed rule on arrival read the null's strict whole-answer gain at step 50. A gain of at least 2.4 points, half the first seed's 4.85, would mean it had arrived by then, along with the switch to "We" openings. A gain under one point would mean it arrived after step 50, and anything between is no call. The first seed reads +1.67, no call. The second seed reads +0.30, "after step 50". The rule looks at one step: at step 20 the same quantity reads +5.99 and +4.32. We do not explain the difference between the seeds' labels.

We expected RL's lenient whole-answer gain at step 10 to stay within a point of zero on the first seed. It read −1.97 (1.04), so that expectation was missed. The second seed, which the rule does not cover, reads +0.83 (1.03).

| greedy answers, GSM8K test: most common opening, count | step 10 | step 20 | step 50 | step 100 |
|---|---|---|---|---|
| null, first seed | Let's break, 615 | We are, 724 | We are, 1,220 | We are, 1,023 |
| null, second seed | Let's break, 966 | Let's break, 1,175 | Step 1:, 956 | Step 1:, 1,319 |
| RL, first seed | Step 1:, 657 | Let's break, 766 | Let's solve, 1,116 | Let's break, 991 |
| RL, second seed | Let's break, 194 | Let the, 47 | There are, 39 | There are, 30 |

**Forcing the opening, first seed only.** Gains are over the unchanged model on the same card, whole answers, GSM8K test. The control passed first: the unchanged model prefilled with its own first two words reproduces its answers on all 1,319 problems.

| state | strict | format-relaxed | lenient |
|---|---|---|---|
| unchanged model prefilled "We are" | −0.83 (1.17) | +1.74 (1.06) | +2.50 (1.02) |
| unchanged model prefilled "We need" | +1.36 (1.12) | +1.21 (1.07) | +1.06 (1.05) |
| null at step 100 | +4.78 (1.23) | +4.17 (1.14) | +4.09 (1.13) |
| null at step 100 prefilled the unchanged model's own openings | −1.06 (1.33) | +2.50 (1.13) | +2.27 (1.12) |

**Reading it.** The two filed tests point opposite ways, and we leave that standing. Giving the unchanged model the null's most common opening did not raise its strict accuracy: the rule says the opening does not carry the null's gain if this reads zero or below, and it reads −0.83. Taking the opening away from the null did lower it: the rule says the opening carries the gain if the null loses at least half of 4.85, and it loses 5.84 (1.12). Unfiled, under the format-relaxed grader the same removal loses 1.67, under a third of what the strict grader shows. No forced-opening read exists for the second seed, whose null never opens "We".

Literature: the scoping report records from Spurious Rewards that random rewards gave 21.4 of 29.1 points on Qwen2.5-Math-7B through clipping, with no gain without clipping. Here there is no clipping, since the ratio is 1, and the null still gains on whole answers: 4.85 on the first seed, and 2.20 with se 1.14 on the second. With an answer past the window counted as wrong, it does not. This is another model, task and grader, and that paper was read only through the scoping report's summary. A later abstract reports Qwen3-8B improving under incorrect rewards.

**Do the updates share a direction?** Hardly. Cosines over all 196 matrices, `results/lr1e-6w10/cosines.json`:

| compared | step 1 | step 10 | step 50 | step 100 |
|---|---|---|---|---|
| RL's update with the null's | 0.085 | 0.036 | 0.004 | 0.015 |
| SFT's with RL's | | | | 0.026 |
| SFT's with the null's | | | | 0.003 |

Step 1 shares its rollouts, which is why it reads highest. So the null's gain is not a leak of RL's update. Nor does the weights' direction distinguish the arms:

| keeps its direction | RL | null | SFT |
|---|---|---|---|
| cosine of step 50's update with step 100's | 0.76 | 0.76 | 0.74 |
| the same, if later steps were orthogonal | 0.69 | 0.67 | 0.71 |
| cosine of step 10's update with step 100's | 0.37 | 0.37 | 0.35 |
| the same, if later steps were orthogonal | 0.20 | 0.18 | 0.21 |

All three keep a bit more than orthogonal steps would, and Adam's momentum supplies some of that. RL's signal is therefore a small part of its weight change, and a size or direction measure alone would not find it.

## Question 4: does SFT stay near the unchanged model because its samples are fixed?

Figure: `figures/kl.png`.

| step | RL: size, KL, KL/size² | null: size, KL, KL/size² | SFT: size, KL, KL/size² |
|---|---|---|---|
| 1 | 0.0035, 0.00018, 15.0 | 0.0035, 0.00005, 4.1 | 0.0036, 0.00026, 20.3 |
| 2 | 0.0079, 0.0011, 18.3 | 0.0076, 0.00015, 2.6 | 0.0075, 0.00011, 2.0 |
| 5 | 0.027, 0.020, 26.5 | 0.025, 0.0015, 2.4 | 0.023, 0.00029, 0.54 |
| 10 | 0.069, 0.064, 13.7 | 0.065, 0.0089, 2.1 | 0.058, 0.00059, 0.17 |
| 20 | 0.136, 0.102, 5.5 | 0.133, 0.072, 4.0 | 0.113, 0.0013, 0.10 |
| 50 | 0.243, 0.184, 3.1 | 0.247, 0.164, 2.7 | 0.194, 0.0009, 0.024 |
| 100 | 0.351, 0.194, 1.6 | 0.370, 0.321, 2.3 | 0.274, 0.0017, 0.023 |

**What the numbers show.**

- The null's divergence stays proportional to its size squared, flat near 2 to 4.
- SFT starts as high as RL, then falls almost a thousandfold on that ratio.
- SFT's divergence fell twice while its size grew, from step 1 to 2 and from step 20 to 50. KL here is on fixed held-out samples with no sampling noise, so these are real moves back in output.
- SFT's weights do not walk back. Its update keeps its direction as the other two do, and its size at step 100 is 0.74 of the null's. So whatever holds the output near the unchanged model acts on what the output is sensitive to, not on the weights as a whole.

**What they don't show.** SFT differs from the null in three ways at once. Its samples are fixed, it has no baseline so every weight is positive, and it trains on twice the tokens. The numbers cannot separate these. RL's ratio also falls, by sixteen times, with samples from current weights, so a falling ratio is not unique to fixed samples.

The candidate has a theoretical basis, stated here as reasoning rather than measurement. Training on the unchanged model's samples minimises divergence from it, up to the tilt toward right answers, while any arm's expected score on its own samples is zero. **The available numbers are what this candidate predicts, and they do not establish it.** Two arms would separate it: shuffled weights on the fixed samples, and SFT on samples from current weights. The second is what the scoping report called online rejection sampling; the plan fixed the samples.

## Question 5: two loose ends

**The factor of ten in the early signal at step 10.** Figure: `figures/delta_r.png`. Partly explained, and the remainder is listed as unexplained.

| state at step 10 | ΔR | KL | size | readout problems | card, rows |
|---|---|---|---|---|---|
| null, no warm-up, probe | 5.78 | 0.190 | 0.138 | 22 | L40, 301 |
| null, warm-up, this run | 0.56 | 0.0089 | 0.065 | 52 | H100, 512 |
| RL, no warm-up, probe | 3.98 | 0.147 | | 22 | L40, 301 |
| RL, warm-up, probe | 2.42 | 0.041 | 0.067 | 22 | L40, 301 |
| RL, warm-up, this run | 5.81 | 0.064 | 0.069 | 52 | H100, 512 |

Three things differ between the null's two readings: warm-up, the readout problems, and the rollouts, since row count changes the draws. Same-recipe RL read 2.42 and 5.81 between these two settings, so problems and rollouts alone move ΔR by more than double. At equal size, step 20 of this run reads ΔR −0.58 against the probe's 5.78. So warm-up is not only a slower clock: no warm-up takes a larger first step, and it reaches a KL of 0.19 where the warmed-up null reads 0.07.

The null's own readings go 0.56, −0.58, 0.68, then 5.10. So ΔR under no signal is not small. The only standard errors we have are from the sanity job: about 1.0 at one step on 11 problems, and 1.3 on 25. None exists at step 10 of either run, so we cannot say how many standard errors separate the two. **Unexplained remainder: how much is noise.**

**Sanity SFT's 0.136 against its rerun's 0.392.** Explained as noise. The two readings had different rollouts. The job's memory set 301 sampling rows in one and 356 in the other, which changes the draws, and the training tokens were 74,562 against 75,601. The readings' standard errors are 0.200 and 0.155. The difference is 0.256 with a standard error of 0.253. The update size agrees to four figures. Neither reading passes that check's five standard errors, which remains its verdict.

## What we learned

Under the strict grader, RL's update raised greedy accuracy on fresh problems more than a shuffled-reward run did on both seeds. How much of that lead is the reward's answer format depends on the seed. On the first seed, on answers all three states finish, format may be most of it, and about one point remains within noise. If an answer past the window counts as wrong, five and a half remain there. On the second seed none of the lead is format, since both arms always write the phrase, and six points remain on every pool. Better maths is not shown on either: a shuffled-reward run gained nearly as much with format set aside on the first seed, and our graders do not check the working. The spread between seeds depends on the pool, from about five points on the filed pool to about one where an answer past the window counts as wrong, and it is unexplained. The null cannot be read as an update that changes no behaviour. On both seeds it rewrites how nearly every greedy answer opens, on the first seed it moves the policy further than RL and loses the answer line for twenty steps. On the second its KL is below RL's and its reward shows no such collapse.

For the decomposition, all three updates are usable, and the second seed's RL and null updates are saved at the same steps. RL against the null compares two large changes in behaviour, one trained on real rewards. SFT is a small change in output for a similar change in weights, so it tests how far weight size alone predicts disturbance. Greedy strict accuracy at one step is a poor measure of what an update did here, and the format-ignored share of samples inside the window is a much steadier one.

## How compelling is the evidence

Two seeds per arm for RL and the null, one for SFT, with greedy answers on two sets at step 100 and on GSM8K test at steps 10, 20 and 50. Standard errors over problems hold the weights fixed, so they say nothing of the spread between seeds. Two seeds give one difference, which sizes that spread from two points only. The null moves about four points between saved steps, so step 100 is one point on a moving curve. On the first seed the sampled lead also moves from about zero to about six points in five-step bins. Sampled readings are on training-pool problems, each new when drawn, with standard errors over steps. The second seed's zero line is the first seed's SFT rollouts, the same prompts with other draws. ΔR is on 52 problems and has no saved standard error.

**What each measure is blind to.** Accuracy is blind to anything between saved states. "Lenient" takes the last number and "format-relaxed" takes the stated number, so both are coarse stand-ins for correct maths and blind to the working. The format-relaxed grader also misses display forms at saved steps, so it still holds format there. The filed rule for a replicating lead reads one seed and ignores the other. The filed rule on the null's gain has no standard error. The arrival rule reads one step. Each forced-opening test changes one state on one seed, and it cannot say what else in the update does the work.

## Surprises

**Something unexpected happened that I cannot explain.** We filed a lean that the format-relaxed lead would come back ambiguous or absent on the second seed, as the first seed's +1.25 suggested. It read +6.42 (1.07), the same as the strict lead, and the filed rule labels it "replicates". The lean reversed. The first seed's lead did not change, so on that pool the two seeds differ by about five points, and we cannot say why. If an answer past the window counts as wrong, a pool read after the fact, they differ by about one.

**Also unexplained.** We predicted the shuffled-reward null would stay within a point and a half. On the first seed it gained 5.6 points on GSM8K test under the drop rule, 4.7 standard errors from zero. The miss stands. Most of it cancels when an answer past the window counts as wrong, and the whole-answer gain does not. On the second seed it gained 1.6 points with a standard error of 1.1, a narrow miss of the same band. No cause of the null's gain is established on either seed.

- The first seed's expected step-10 reading was missed: RL's lenient gain read −1.97, against within a point of zero.
- The two forced-opening tests read opposite ways on the first seed.
- The second seed's null moved every greedy answer to one opening, "Step 1:", and not to "We", while RL's second seed shows no common opening.
- SFT's KL from the unchanged model is about 113 times smaller than RL's, against a filed "at least three times larger". That reverses the filed direction.
- On the first seed, the null's sampled reward collapsed by about nineteen points for twenty steps, by losing the answer line, and recovered.
- On the first seed, every greedy answer of the null moved to two opening words.
- ΔR under the null reached 5.1 at step 100, failing the filed margin there.

## What next

1. **The decision below on a third seed.** A third lead adds a point on each pool. Which seed then looks odd depends on the pool, since the two seeds differ by five points on one and by one on another.

Also $0, for the question on SFT: shuffled weights on SFT's saved rollouts, about 1.5 h of an L40. If KL stays near 0.002, fixed samples are enough to hold the output.

## Pending decision

*Question for you · `qwen-rl-lead` · new 2026-11-04. Urgency: Soon. Due to run: nothing until this is decided; the third seed runs only if chosen.*

**Should the reinforcement-learning row of the update table rest on the two seeds we have, or wait for a third for about $20?**

The update table compares how a decomposition reads different kinds of training update. Its reinforcement-learning row needs to say how much the training beat the same loop with rewards shuffled. We have run that comparison twice from the unchanged model. Under the exact-format grader the training leads on the same test problems by about three and six points. With format set aside, the second run stays near six on every treatment of long answers. The first run reads about one point on answers that finish, two and a half on whole answers, and five and a half if an answer past the window counts as wrong, as the reward counts it. So the gap between runs is five points, four or one, depending on that treatment. Nothing we have measured explains why it depends on it. The training's own gain is close between runs. Neither run shows better maths.

- **A. Read the two seeds as they stand.** Nothing more runs, at $0. The row reports both leads on all three treatments, and the decomposition reads the saved updates from both runs. It tells us the training beat the shuffled loop twice under the strict grader, with a size we cannot state. Next, the decomposition work goes ahead now. Its limit is that any later claim leaning on the size of the lead stays open.
- **B. Run a third seed first.** The training and the shuffled loop run again on one rented H100 for about 5 hours and about $20, followed by about 2 hours of free cluster time for the step reads. It adds a third lead on each treatment of long answers. Next, the row reports three leads, a day later. Its limit is that three runs give only a rough spread, and which run looks odd may differ by treatment.

**Recommendation: A.** Read the two first. One reading does depend on these runs: our plan uses the shuffled loop's gain as the noise floor for reading the training's update, and that gain differs by run, so we use the larger value, the first run's. We know of no other decomposition reading that yet turns on the lead's exact size, and the third seed stays available if one comes to.

*Names in the reports: the update table is D2; the two runs are E69 and its Amendment 1. "Seed 1" and "seed 2" are labels: seed 1 is draw seed 69 and seed 2 is draw seed 70.*

## Scorecard

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| P0 | the unchanged model's accuracy near the baseline, readings zero at step 0 | within 1.5 points of 77.7 and 78.2; ΔR and KL ≤ 1e-6; non-empty pool | 78.17 and 78.89; 0 and 0; 52 problems | holds |
| P1 | RL learns | gain ≥ 2 points and ≥ 2 SE; under 1 point no gain | +9.04 (1.13) | learns |
| P2 | not memory | GSM1K minus GSM8K: under 2 points and within 2 SE | −1.96 (1.64) | same |
| P3 | the null stays flat | abs(gain) < 1.5 points | +5.63 (1.19) | **miss**: 4.1 points past the band |
| P4 | early signal | RL's ΔR > 0 at every saved step; ≥ 3 × abs(null's) at 10, 50, 100 | positive at all; 10.3×, 18.5×, then 1.7× (8.69 against 5.10) | **miss** at step 100 |
| P5 | withdrawn before filing | | | |
| P6 | length steady | steps 91-100 within ±25% of steps 1-10; share in window ≥ 0.90 | 1.135×; 0.963 | holds |
| P7 | SFT moves the policy more, and gains no more | F's KL(100) ≥ 3 × RL's; F's gain ≤ RL's | 0.0017 against 0.194; +2.58 against +9.04 | **miss**: first part reversed, second holds |

"Learns" is the filed label for a gain that met P1's rule; it does not say what the gain consists of. Branch taken: R1, on that label. The tree's P3 line applies: an FYI, the null's δW still read, and RL's reading carries the gap as its noise floor. That noise floor varies by seed, 5.63 and 1.62 on the filed gain, and we use the larger. The tree files no rule for RL minus the null. The difference of the two filed gains is 3.41, and the same-problem lead on the three-state pool, chosen afterwards, is +3.2 with se 1.0.

**Amendment 1**, `results/predictions_amend1.json`, scored in `results/amend1_score.json`.

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| G0 | control: the format-relaxed grader reproduces the first seed's lead | +1.25 (0.90) and +0.81 (1.06) as printed | both as printed | holds |
| S0 | control: the second seed's RL learns, by P1's rule | gain ≥ 2 points and ≥ 2 SE | +8.31 (1.11) | learns; the filed label, not a claim about maths |
| S1 | RL's lead on the second seed, strict; lean: replicates | ≥ 2 points and ≥ 2 SE: replicates; < 1: no lead; else ambiguous | +6.42 (1.07), 1,200 problems | replicates |
| S1r | as S1, format-relaxed; lean: ambiguous or no lead | as S1 | +6.42 (1.07), 1,200 problems | replicates; **lean reversed** |
| S2 | the null's gain on the second seed is systematic | abs(gain) ≥ 1.5 with the first seed's sign; < 1.5: one-seed walk | +1.62 (1.11) | systematic, 0.12 past the band |
| saved-step control | the unchanged model on the job's card; the null rebuilt at step 100 | W0 gap under 1.5 points; flips of the null's answers ≤ 3 per set (amended 2026-11-04, after one flip stopped the job, before any state it guards) | W0 gap 0, 0 flips; 1 of 1,319 and 0 of 1,197 flips | holds |
| B1 | the null's gain arrives by step 50, whole answers, strict, each seed alone | ≥ 2.4: arrives; < 1.0: after step 50; else no call | first seed +1.67; second seed +0.30 | no call; after step 50 |
| E_R10 | expected: RL's lenient gain at step 10, first seed | abs(gain) ≤ 1 | −1.97 (1.04) | **missed** |
| openings control | the unchanged model prefilled its own opening | same answers, or within a batch floor | 0 of 1,319 differ | holds |
| A1 | "We are" on the unchanged model carries the gain, strict; expected: not | ≥ 2.4: carries; ≤ 0: not; else part | −0.83 (1.17) | not |
| A2 | the null without its opening loses the gain, strict | loss ≥ 2.425: carries; else not | 5.84 (1.12) | carries |

Unfiled, as the rules say: the spread between seeds, the relaxed and drop-rule columns of saved steps and openings, "We need", and GSM1K.

## Departures from the literature

- **The second seed, the saved-step reads and the forced openings add none:** the recipe is the first seed's. The format-relaxed grader and the prefilled openings are our own measures, credited to no source.

- **Dr. GRPO form.** As recorded in the scoping report: a group-mean baseline, no standard-deviation scaling, loss summed over tokens and divided by a constant. Our constant is 64 · 8 · 512. The paper's batch and rate were not checked against ours here; they came from the 2026-10-28 plan.
- **One update per batch, so no ratio clipping and no KL penalty.** GRPO-style trainers usually have both. Chosen to keep the update the plain policy gradient and remove the clipping effect of Spurious Rewards.
- **Warm-up,** ten linear steps to the full rate. The 2026-10-28 recipe said none. Without it, the sanity job's first SFT step gained about nothing.
- **Full-precision forward,** not bf16 autocast as real post-training runs use. Autocast's rounding would keep 1e-6 steps at the unchanged model.
- **Own samples fixed at the unchanged model** in SFT. The scoping report's matched arm sampled from current weights. This changes what SFT means; see Question 4.
- **The null shuffles within each problem,** where Spurious Rewards used random rewards. It keeps each problem's reward distribution exactly.
- **Reward requires ending inside the window,** a constraint from the decomposition, not from any cited recipe.
- **Greedy strict accuracy at the end only,** with no best-checkpoint selection, so the null's reading is one point of its walk.

## Not run

- A third seed of RL and the null, and a second seed of SFT.
- A recomputation of the δW hashes by the independent check; the fetch checked each file against the run's checksums (job 51098).
- Greedy accuracy at steps 1, 2 and 5, or on GSM1K at any saved step; nothing was saved inside the first seed's collapse.
- Forced openings on the second seed, or on GSM1K.
- The accounting by window class, the length tests and the cosines on the second seed.
- An arm varying one difference between SFT and the null: fixed samples or no baseline.
- A standard error on ΔR at any step of this run, or a read of the probes' states on this run's problems.
- Per-token entropy of any arm's policy.
- Correct maths graded more finely than the last number.
- Autocast rollouts. Any map or read of the decomposition, which is `qwen-decomp-lead`'s next.

## Provenance

- Results: `experiments/qwen_rl/results/lr1e-6w10/{R,N,F,score}.json`, `run.log`, `N_stopped_kl_band.json`; `results/{A0,S,sanity}.json`; `results/probe_*.json`.
- Large files: `results/large/acc/W0_*.jsonl`, `results/large/lr1e-6w10/acc/`, `dW/`, `rollouts/{R,N,F}.tar`. The unchanged model's two files and all three tars match the checksums in `results/`.
- Numbers and figures: `experiments/qwen_rl/plot.py`, `rollouts.py`, `cosines.py`, on CPU, with no model loaded. Token ids checked against the tokenizer: 8304 "Step", 10061 "Let", 1654 "We".
- One H100, $33 in all. No GPU was used for this report.
- Amendment 1. Second seed: `results/lr1e-6w10_s70/{R,N}.json`, `run.log`, `SHA256SUMS`; per-problem answers in `~/commute-data/results-large/qwen_rl/lr1e-6w10_s70/acc/`. Saved steps and openings: `results/{lr1e-6w10,lr1e-6w10_s70}/amend1/`, `amend1.log`. Second seed's decoded samples: `rollouts.py`, `results/lr1e-6w10_s70/rollouts_summary.json`. Scores: `score_amend1.py`, `relaxed.py`. Readings and figures `leads.png`, `saved_steps.png`: `plot_amend1.py`, on CPU, with no model loaded.
- Amendment 1's compute. One RunPod H100: the two arms took 3.9 h by the run log, against the plan's estimate of about $20. The bill is not in these files. Cluster `debug` on an L40, $0: jobs 50942-50952, 51005, 51027-51034 for the first seed, and 51099-51197 for the second seed's saved steps, of which several are resubmissions (`cluster/logs/jobs.log`).
