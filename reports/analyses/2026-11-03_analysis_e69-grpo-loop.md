# Does RL on Qwen3-1.7B gain more than the same loop with shuffled rewards?

*E69, 2026-11-03. Plan `reports/plans/2026-11-03_plan_e69-grpo-loop.md`, notebook `notebooks/E69_qwen_rl.ipynb`, numbers
`experiments/qwen_rl/results/lr1e-6w10/{analysis,rollouts_summary,cosines}.json`, figures `experiments/qwen_rl/figures/`. Wording of the claims revised the same day after `reports/reviews/2026-11-03_review_e69-independent-check.md`; no filed prediction changed.*

We trained Qwen3-1.7B on maths problems by reinforcement learning, by the same loop with rewards shuffled, and by supervised training on its own correct answers. Reinforcement learning beat the shuffled loop on the same test problems when the exact answer format was required, and on new problems as it trained, while the shuffled loop's own gain mostly cancels once answers running past the window count, for a reason we have not established. The updates are ready to go forward to the decomposition, with the shuffled loop read as a large change in behaviour without signal and not as a do-nothing.

## The answer

Under the strict grader, RL beat this shuffled-reward run by three to eight points on the same test problems, depending on how long answers are counted. On answers all three states finish it reads +3.2 and +3.1. The standard error over problems, with the weights fixed, is near one point. Each arm is one seed, so the spread between runs is not measured, for either arm. With format ignored, on those same answers, the lead is about one point, +1.2 and +0.7, and not separable from zero. Accepting forms like "So the answer is 20" takes the strict lead there to +1.25 (0.90) and +0.81 (1.06). So the firm lead may be largely answer format, and better maths is not shown. The null's own gain is also one seed's and remains unexplained.

**The claim we can make, in plain words.** Under the strict grader, RL's update raised accuracy on unseen problems more than this shuffled-reward update did, on both test sets, under every way of treating long answers. The lenient grader gives the same sign in every cell, but only some of those leads stand clear of noise. The lead is largest when an answer past the window counts as wrong, as it does for the reward, and most of it there is the null's answers not finishing. With format ignored on that rule, RL leads by four to six points. On sampled answers read the same way, RL climbs steadily to five points above the unchanged model, with a standard error of 1.1 to 1.2, while the null stays within two points of it. RL's sampled lead over the null reads +5 in the last five steps and +3 averaged over the last fifty. The null's right-number share among finished samples rises as much as RL's, and its lower finishing share cancels that, subject to the selection noted below.

**The claim we cannot make yet** is that RL does better maths than this null, or that the lead holds against shuffled-reward runs in general. RL's gains of nine and seven points overstate what the reward added, since format is part of them and the null gained too. With format ignored, the lead is about two points if answers may run long, and about one, within noise, on answers all three states finish. Keeping a problem because the null finished it may favour the null, so that one point may understate RL. Kept on the unchanged model alone, whole answers read +2.15 and +1.27, so the right-number lead is one to two points unless failing to finish counts against an answer.

| RL minus null, same problems, points (SE) | drop rule | whole answers | capped answers |
|---|---|---|---|
| GSM8K test, strict grader | +3.2 (1.0) | +4.6 (1.0) | +7.4 (1.1) |
| GSM1K, strict grader | +3.1 (1.2) | +4.8 (1.2) | +8.4 (1.2) |
| GSM8K test, lenient grader | +1.2 (0.9) | +2.1 (1.0) | +4.3 (1.0) |
| GSM1K, lenient grader | +0.7 (1.1) | +2.4 (1.1) | +6.3 (1.2) |
| GSM8K test, lenient, kept at the unchanged model only | | +2.15 (0.96) | |
| GSM1K, lenient, kept at the unchanged model only | | +1.27 (1.19) | |
| GSM8K test, strict, other forms of the answer line accepted | +1.25 (0.90) | | |
| GSM1K, strict, other forms of the answer line accepted | +0.81 (1.06) | | |

The drop-rule column keeps problems that fit at the unchanged model, RL and the null alike: 1,123 and 868. The row kept at the unchanged model only has 1,255 and 1,023 problems and no selection on the null's outcome. The last two rows also accept the lower-case phrase and the `\boxed` and bold wrappers. All four added rows come from `reports/reviews/2026-11-03_review_e69-independent-check.md`, not from `analysis.json`. These standard errors count problems at fixed weights. They carry no seed-to-seed spread of either arm, which this run cannot measure. The two test sets are two samples of problems read from the same two weights, so their agreement does not replicate the arm difference.

## What we were trying to test

The project needs real post-training updates to carry through a decomposition, and no reinforcement-learning update of this model existed. We trained the unchanged model, with thinking off, for one hundred steps on GSM8K train, three ways, on the same prompts at each step.

- **RL.** 64 new problems per step, 8 samples each at temperature 1. Reward is 1 for a strictly right answer that ends inside the 512-token window, else 0. Weight is reward minus the problem's mean reward.
- **The shuffled-reward null.** The same, sampling from its own changing weights, with rewards shuffled among each problem's samples. It rules out that any update of this size moves accuracy by itself.
- **Own-samples SFT.** Weight 1 on the unchanged model's correct samples, drawn once and never redrawn, on the same prompts.

All use AdamW at a rate of one in a million, reached over ten warm-up steps. Accuracy is the greedy answer. The strict grader wants the final line "The answer is N."; the lenient grader takes the last number.

Long answers need a rule, and `qwen_eval.summary` defines three. The **drop rule**, the filed one, keeps problems whose answers fit the window at both states. **Whole answers** grades every problem on its full response, up to 512 new tokens. **Capped answers** grades every problem on the text the window keeps.

## What we expected, and why

We expected RL to gain three to eight points on test and about the same on GSM1K. We expected the null to stay within a point and a half, because its expected gradient is zero. One update per batch also removes the clipping that the scoping report, citing Spurious Rewards, blames for gains under random rewards. We expected the right-over-wrong signal to grow in RL only. Whether SFT moves the policy more than RL was left open.

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

**Reading it.** RL gains under every rule, and the rule barely changes its number. The null ranges from a clear gain to nothing as long answers count against it. SFT's strict gain is almost all format, since its lenient gain is near zero on both sets. On GSM8K test, no RL answer that fits the window is right yet missing the answer line, while the unchanged model has 29.

## Question 3: what the null's gain consists of

Figures: `figures/accounting.png`, `figures/openings.png`, `figures/lengths.png`.

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

**What changed in the answers.** All of the null's greedy answers on both sets open "We are" or "We need". At the unchanged model 19 do on GSM8K test, and none of the null's answers is identical to the unchanged model's. RL's answers moved another way: "Step 1:" went from 425 to zero and "Let's" took over. SFT stayed close, with "Step 1:" rising to 540 and 420 answers identical word for word.

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
5. *The opening flipped, and the new opening is better.* This was our favoured hypothesis, and its most direct test counts against it. In the unchanged model's own 51,200 samples, "We" openings can be compared with "Let" openings on the same problem, across 644 problems. Ignoring format and counting only answers inside the window, "We" is 2.5 points worse with se 1.2. It is also 9.5 points worse on reward with se 1.5, and 20 tokens longer. "Step" against "Let" on the same problem reads 0.0 with se 0.7, so the large differences between openings in the table above are which problems each opening goes with. The test reads samples of the unchanged model, not greedy answers of the null, so it does not rule out an opening effect that only appears there.
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

**How firmly it stands.**

- The accounting is arithmetic on the per-problem files, and it stands.
- The collapse being format stands, from the decoded text of one run.
- The null's real whole-answer gain, 4.85 with se 1.23, stands as a reading of this seed at step 100.
- Its cause is unexplained: we can rule out format, selection, and length problem by problem. The opening stays open, with its most direct test counting against it.

Literature: the scoping report records from Spurious Rewards that random rewards gave 21.4 of 29.1 points on Qwen2.5-Math-7B through clipping, with no gain without clipping. Here there is no clipping, since the ratio is 1, and the null still gains on whole answers. With an answer past the window counted as wrong, it does not. This is another model, task and grader, and that paper was read only through the scoping report's summary. A later abstract reports Qwen3-8B improving under incorrect rewards.

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

Under the strict grader, RL's update raised accuracy on fresh problems more than a shuffled-reward run did, sampled and greedy. Part of that lead is the reward's answer format, and on answers all three states finish that part may be most of it. With format ignored and an answer past the window counted as wrong, four to six points of greedy lead remain, most of them because the null's answers more often fail to finish. On answers all three states finish, about one point remains, within noise, so better maths is not shown. The null cannot be read as an update that changes no behaviour. It moves the policy further than RL, rewrites how every greedy answer opens, and can lose the answer line for twenty steps.

For the decomposition, all three updates are usable. RL against the null compares two large changes in behaviour, one trained on real rewards. SFT is a small change in output for a similar change in weights, so it tests how far weight size alone predicts disturbance. Greedy strict accuracy at one step is a poor measure of what an update did here, and the format-ignored share of samples inside the window is a much steadier one.

## How compelling is the evidence

One seed per arm, one accuracy reading at step 100, greedy answers on two sets. Standard errors over problems hold the weights fixed, so the spread between runs is not measured, for either arm. The sampled lead moves a lot between steps, from about zero to about six points in five-step bins, so step 100 is one point on a moving curve. Sampled readings are on training-pool problems, each new when drawn, with standard errors over steps and shared prompts and seeds. ΔR is on 52 problems and has no saved standard error. Accuracy is blind to anything between saved states. "Lenient" takes the last number, so it is a coarse stand-in for correct maths.

## Surprises

**Something unexpected happened that I cannot explain.** We predicted the shuffled-reward null would stay within a point and a half. It gained 5.6 points on GSM8K test under the drop rule, 4.7 standard errors from zero. The miss stands. Most of it cancels when an answer past the window counts as wrong, and the whole-answer gain does not.

- SFT's KL from the unchanged model is about 113 times smaller than RL's, against a filed "at least three times larger". That reverses the filed direction.
- The null's sampled reward collapsed by about nineteen points for twenty steps, by losing the answer line, and recovered.
- Every greedy answer of the null moved to two opening words.
- ΔR under the null reached 5.1 at step 100, failing the filed margin there.

## What next

1. **Force the opening; $0, about an hour of a cluster L40.** The unchanged model's greedy answers prefilled "We are" and "We need" on GSM8K test, and the null's prefilled with the unchanged model's own opening. If the prefilled unchanged model reaches the null's accuracy, the opening carries the gain. If not, as the sampled test suggests, the rest of the update does.
2. **Greedy accuracy at the null's saved steps 20 and 50; $0, about an hour of a cluster L40.** If the whole-answer gain tracks the "We" flip, it should be present by step 50. If it arrives only at step 100, one reading of a walk is what we have.
3. **A second null seed.** About 2.3 h of H100, about $9, or $0 on an L40 if the queue is empty and its memory limit is handled. Its gain sizes RL's lead: none means RL's whole-answer lead is its full gain, and as much means the lead stands as reported.

**Planned, not yet run:** `reports/plans/2026-11-03_plan_e69-amend1.md` takes up items 1 and 2 and replaces item 3 with a second seed of both RL and the null, read on a strict and a format-relaxed grader.

Also $0, for the question on SFT: shuffled weights on SFT's saved rollouts, about 1.5 h of an L40. If KL stays near 0.002, fixed samples are enough to hold the output.

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

"Learns" is the filed label for a gain that met P1's rule; it does not say what the gain consists of. Branch taken: R1, on that label. The tree's P3 line applies: an FYI, the null's δW still read, and RL's reading carries the gap as its noise floor. The tree files no rule for RL minus the null. The difference of the two filed gains is 3.41, and the same-problem lead on the three-state pool, chosen afterwards, is +3.2 with se 1.0.

## Departures from the literature

- **Dr. GRPO form.** As recorded in the scoping report: a group-mean baseline, no standard-deviation scaling, loss summed over tokens and divided by a constant. Our constant is 64 · 8 · 512. The paper's batch and rate were not checked against ours here; they came from the 2026-10-28 plan.
- **One update per batch, so no ratio clipping and no KL penalty.** GRPO-style trainers usually have both. Chosen to keep the update the plain policy gradient and remove the clipping effect of Spurious Rewards.
- **Warm-up,** ten linear steps to the full rate. The 2026-10-28 recipe said none. Without it, the sanity job's first SFT step gained about nothing.
- **Full-precision forward,** not bf16 autocast as real post-training runs use. Autocast's rounding would keep 1e-6 steps at the unchanged model.
- **Own samples fixed at the unchanged model** in SFT. The scoping report's matched arm sampled from current weights. This changes what SFT means; see Question 4.
- **The null shuffles within each problem,** where Spurious Rewards used random rewards. It keeps each problem's reward distribution exactly.
- **Reward requires ending inside the window,** a constraint from the decomposition, not from any cited recipe.
- **Greedy strict accuracy at the end only,** with no best-checkpoint selection, so the null's reading is one point of its walk.

## Not run

- A second seed of any arm.
- Greedy accuracy at any step before 100, although states are saved at steps 1, 2, 5, 10, 20 and 50; nothing was saved inside the collapse.
- Answers forced to a chosen opening, at the unchanged model or the null.
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
