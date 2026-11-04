# Independent check of E69: did RL beat the shuffled-reward null on unseen problems?

*2026-11-03. Computed from the per-problem files before reading the analysis. Scripts and full outputs:
`~/commute-data/scratch/qwen-rl-lead/e69-check/` (`own_numbers`, `more`, `third`, `fourth`, each `.py` and `.out`). Claim
checked: `reports/analyses/2026-11-03_analysis_e69-grpo-loop.md`. CPU only, no model loaded.*

Every number I checked in the analysis reproduces, and I found no bug in the grading, the pairing or the problem sets. RL's lead over the null is real under the strict grader, but one claim sentence and one caveat are wrong as worded. On answers that all three states finish inside the window, with format ignored, the lead is about one point and not separable from zero, so the lead does not show RL getting more numbers right than this null on those answers.

## My answer to the question

RL raised accuracy on unseen problems more than this one shuffled-reward run, by 3.2 to 8.4 points under the strict grader. The standard error over problems is 1.0 to 1.2 in every strict cell. What that lead consists of depends on what is counted:

| RL minus null, points (SE over problems); z | GSM8K test | GSM1K | pooled |
|---|---|---|---|
| strict, kept at all three states | +3.21 (0.99); 3.2 | +3.11 (1.15); 2.7 | +3.16 (0.75); 4.2 |
| lenient, kept at all three states | +1.16 (0.89); 1.3 | +0.69 (1.06); 0.65 | +0.95 (0.68); 1.4 |
| lenient, kept at the unchanged model only (not in the analysis) | +2.15 (0.96); 2.2 | +1.27 (1.19); 1.1 | |
| lenient, whole answers | +2.12 (0.95); 2.2 | +2.42 (1.14); 2.1 | +2.27 (0.74); 3.1 |
| lenient, capped answers | +4.25 (1.01); 4.2 | +6.27 (1.19); 5.3 | |
| strict, capped answers | +7.35 (1.08); 6.8 | +8.35 (1.20); 7.0 | |

Against the unchanged model, RL's gain is not in doubt at these weights. It is at least +5.6, z about 5, in every cell of the analysis's gain table. The lowest I could find on any pool is +3.6 (1.17), z 3.1: GSM1K, capped lenient, over problems that fit at the start.

## 1. Does the table reproduce, and are the standard errors right?

**It reproduces, all twelve cells, to the rounding shown.** The strict and lenient drop-rule cells come from the problems that fit at the unchanged model, RL and the null: 1,123 and 868, as stated. The gain table's filed cells match `R.json`, `N.json`, `F.json`, and P2's −1.96 (1.64) matches.

**The standard errors are the right kind for what they measure.** They are the SD of per-problem differences over root n, the paired form, and agree with the discordant counts. Treating the arms as independent would give 1.64 and 1.77, which is wrong for the same problems.

**They answer a narrower question than "how firm".** They hold both sets of weights fixed and resample problems only. Two points follow.

- The caption says they carry no spread of the null. They also carry none of RL's, since each arm is one walk.
- Two test sets are two samples of problems read from the same two weights. Agreement between them is not a replication of the arm difference.

**The size of the unmeasured part is not small, on the evidence available.** Over steps 51-100 the sampled, format-ignored lead of RL over the null, in five-step bins, reads −0.2, 3.0, 3.2, 0.7, 2.3, 2.3, 4.6, 2.3, 6.1, 5.2. The same bins of the reward lead run from 3.1 up to 30.4 and back to 11.1. The null's reward moves from 0.529 at step 88 to 0.764 at step 100. So a strict greedy lead read at one step is one point on a moving curve, and step 100 is not representative of the steps before it in either direction. A second null seed, already on the analysis's list, is what would size this.

**The comparison itself was not filed.** The filed tree says RL's reading "carries the gap as its noise floor", with no rule for RL minus null, and the three-state pool is a choice made afterwards. The analysis's "filed-rule gain over the null is +3.2" is that post-hoc pool. The difference of the two filed gains is 9.04 − 5.63 = 3.41. The choice moves the number by 0.2, so it does not matter to the reading, but the phrase should not say filed.

## 2. Sentence by sentence

**"Yes. On the same test problems RL beat the shuffled-reward null by three to eight points, depending on how long answers are counted."** Follows for the strict grader only. Change to: "Under the strict grader, RL beat this shuffled-reward run by three to eight points".

**"The standard error is near one point."** True and incomplete. Change to: "The standard error over problems, with the weights fixed, is near one point; the spread between runs is not measured."

**"With the answer format ignored and answers held to the window, RL leads by about five, and samples on new problems agree."** This is the sentence that most needs changing, for two reasons.

- "Held to the window" fits two columns. The capped column reads +4.3 and +6.3, but the drop-rule column is also format-ignored and inside the window, and reads +1.2 and +0.7. The sentence reports only the first.
- Most of the five is the null failing to finish, not RL getting more numbers right. In net problems, on the capped lenient rule, problems that fit at both RL and the null supply 15 of 56 on GSM8K test and 6 of 75 on GSM1K. Problems where only RL fits supply +48 and +75, partly offset by the other cells.

The samples say the same. In the last five steps, the share of finished samples with the right number is 0.896 in both arms, and the share finishing is 0.959 against 0.902. "Agree" also holds only for those five steps: the sampled lead is +5.2 there, +3.8 over steps 76-95, +1.8 over 51-75 and +3.0 over 21-50, averaging +3.0 over steps 51-100.

Suggested text: "With format ignored, and an answer past the window counted as wrong, RL leads by four to six points, most of it because the null's answers more often fail to finish. On answers all three states finish, the lead is about one point and within noise. Sampled answers read +5 in the last five steps and +3 averaged over the last fifty."

**"RL's nine points overstate what it learned, since format is part of them and the null gained too."** Follows. Nine is GSM8K test only, and GSM1K reads seven; I would write "nine and seven".

**"That null gain is one seed's and remains unexplained."** Follows, and RL's lead is equally one seed's.

**"RL's update raised accuracy on unseen problems more than this shuffled-reward update did, on both test sets, under every way of treating long answers."** True for the strict rows, each at z of at least 2.7. Read against all twelve cells, the two lenient drop-rule cells have the right sign but are at z 1.3 and 0.65, and pooled at 1.4. The two lenient whole-answer cells are at z 2.2 and 2.1, which pass singly but not after allowing for twelve cells; pooled they read +2.27 (0.74). Add "under the strict grader" and say what the lenient grader shows.

**"The lead is firmest inside the window, which is what the reward and the decomposition see."** True only of the capped column and contradicted by the drop-rule lenient cells, which are also inside the window and are the weakest in the table. Change to: "The lead is largest when an answer past the window counts as wrong, as it does for the reward, and most of it there is the null's answers not finishing."

**"There, on sampled answers with format ignored, RL climbs steadily to five points above the unchanged model while the null stays within two points of it."** Holds on the analysis's spans. I reproduced +0.1, +0.3, +2.5, +3.4, +4.1, +5.0 and +0.2, −0.2, −0.6, +1.6, +0.3, −0.2. Add that the +5.0 has a standard error of 1.1 to 1.2. Its reading is also not "RL only": the null's right-number share among finished samples rises as much as RL's, and its lower finishing share cancels that, subject to the selection the analysis notes.

**"If answers may run long and format is ignored, the lead is only one to two points."** Mislabelled. "May run long" is the whole-answers column, which reads about two (2.1 and 2.4). The "one" comes from the drop-rule cells (1.2 and 0.7), where every answer counted is inside the window. Change to: "With format ignored, the lead is about two points if answers may run long, and about one, within noise, on answers all three states finish."

**"These standard errors count problems at fixed weights. They carry no seed-to-seed spread of the null, which this run cannot measure."** Change "of the null" to "of either arm".

**Also carrying the same claim: line 223,** "about five points inside the window remain with format ignored". Give it the same qualifier. **Line 6's abstract** holds as written.

**A caution on the drop rule, in the analysis's favour.** Keeping a problem because the null finished it conditions on the null's own outcome, which probably favours the null. So +1.2 and +0.7 may understate RL. The row kept on the unchanged model alone, which has no such selection, reads +2.15 (0.96) and +1.27 (1.19) on whole answers, and should be added to the table. It does not change the conclusion that the right-number lead is one to two points unless failing to finish counts against an answer.

## 3. The null's gain: +125 −61 = 64, and 53 past the cut

**The arithmetic holds, all of it.** On GSM8K test the four rows reproduce exactly:

- fits at both, 1,137: +125 −61 under both rules;
- ran long, 118: +19 −30 whole, +0 −53 capped, with 53 right at the unchanged model and 23 of those right if allowed to finish;
- came back, 17: +12 −1 whole, +12 −0 capped;
- fits at neither, 47: +9 −9 whole, +0 −1 capped.

The totals are +64 = 4.85 points and +22 = 1.67 points. GSM1K's +36 and −23, RL's +124 and +119, 4.22, 0.88 and 8.83 all reproduce too.

**Two coincidences that could mislead a reader, not errors.**

- The +64 on the 1,137 kept problems and the +64 on all 1,319 are the same number only because the other rows sum to zero (−11 +11 +0).
- Two different 53s appear. One is the 53 right answers lost past the window, and the other is the net +53 behind 4.22 (64 − 11, over 1,255).

**"The capped number differs for one reason" holds, with one precision.** The whole-to-capped fall is 42 problems, not 53, because that row already reads −11 on whole answers. Only 23 of the 53 are lost to the cut alone, while 30 are wrong even when finished.

## 4. Bugs

**None that moves these numbers.** Each check below came back clean.

- **Grades.** A separate copy of the strict rule agrees with the stored grade on all 10,064 records across the four states and both sets.
- **Window rule.** `fits` recomputed from the token counts has 0 mismatches, and no fitting answer is truncated. A capped answer differs from the whole answer in no fitting case. "Truncated" differs from "512 tokens" in 2 records, answers ending exactly at the cap.
- **Pairing.** Problem indices, gold answers and prompt token counts are identical across all states. Gold answers match `qwen_sets.load()` on both sets.
- **Training problems.** 6,400 distinct, none repeated, none in the held-out set; each step's problems are the same in RL and the null, and step 1's tokens are identical in all three arms.
- **Unseen.** No test or GSM1K question exactly matches a training question used. One GSM8K test problem (index 602) shares two thirds of its word 8-grams with the training problems used, and both arms get it right, so it moves nothing. GSM1K's highest overlap is 0.138.
- **Evaluation path.** The unchanged model's accuracy files from two separate H100 runs have the same checksum, so greedy evaluation repeated exactly. SFT's lenient gain near zero and its 420 word-identical answers show that trained weights go through the same path.
- **Null's first run.** It stopped on the first-step divergence band, which was then widened (`2b3ba97d`). That band is a stop check and is not a scored rule. The rerun's first divergence reading matches the stopped run's to five figures, so it started the same way.

**One grader sensitivity, not a bug.** The strict rule is case-sensitive, as filed and as the prompt's "exactly this form" asks. 29 of the null's 31 finished GSM8K test answers with no answer line, and 26 of its 29 on GSM1K, end in the phrase in another form, such as "So the answer is 20." The unchanged model's missing lines are of another kind: 43 of 50 on GSM8K test wrap the number in `\boxed` or bold. Accepting the lower-case form and those two wrappers moves RL minus null on the three-state problems to +1.25 (0.90) and +0.81 (1.06). So the strict drop-rule lead of about three points is mostly this format difference. Among problems fitting at both RL and the null, it is 24 of 39 net on GSM8K test and 21 of 27 on GSM1K.

## Not run

- Any second seed of any arm, or any model load: the spread between runs remains unmeasured.
- Greedy accuracy at any step before 100.
- The analysis's Questions 4 and 5, the cosine tables, the opening-word tables, explanations 2, 4, 5 and 6, and ΔR, none of which the answer paragraph rests on.
- Checks of correct maths beyond the last number in an answer.
