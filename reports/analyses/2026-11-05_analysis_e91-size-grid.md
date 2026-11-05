# How big is each Qwen update when it matches the template fine-tune, and what has it learned there? (E91)

*D2 row 11-06. 2026-11-05, `qwen-rl-lead`. Plan `reports/plans/2026-11-05_plan_qwen-reorient-amend1.md`, rules `experiments/qwen_size_grid/results/predictions.json`, `predictions_matched.json` and `predictions_past_line.json`, notebook `notebooks/E91_qwen_size_grid.ipynb`.*

We found each Qwen update's saved state closest in size to the template fine-tune's full update, and saved the TOFU facts fine-tune there and at its first state past the learning line. At the template's size the TOFU update recalls its facts like the unchanged model, and its smallest saved state holding them is about 30 times that norm. For `GOALS.md`, the two cannot be compared at equal size with both jobs done, so the matched step is a size control only and any claim of new content rests on the full update, a choice the user called a weak endorsement, not having fully followed the question.

## The answer

The table exists, and the matched step cannot carry the comparison alone. Every arm has a named state nearest the template fine-tune's full update. At that size the TOFU update reads like the unchanged model on its facts. Its smallest saved state holding them is about thirty times that norm. So the comparison reads both: the matched step controls size only, and any claim of new content rests on the full update, with its size stated. The user chose this as a weak endorsement, not having fully followed the question.

## What we were trying to test

The Qwen comparison needs its updates read at a common size, because the TOFU update ends 37 times larger than the template one. The amended plan made the size-matched step the reading the comparison leans on. This job finds that step for each arm, asks whether the TOFU update has learned anything by then, and then asks how large it is once it has.

**The reference** is anchor 1's full update at its first seed, step 100. Anchor 1 is the template fine-tune, which teaches an answer format the model already follows under a prompt. Its norm is 0.4865 and its KL is 0.0959.

**Two sizes are read on each saved state:**

- **Norm:** the Frobenius norm of the update over the decomposed matrices. The unchanged model reads 0.
- **KL to start:** $\mathrm{KL}(W_0 \,\|\, W_0 + \delta W)$ per position on the first 16 maths rows, in full precision. The unchanged model reads 0. These are the rows anchor 1's training tracked.

"Nearest" is the smallest absolute log ratio to the reference. An arm that never reaches the reference gets its largest state, flagged as not reached.

**The arms, and what each rules out or adds:**

- **Anchor 1, two seeds.** The second seed shows how far a repeat lands from the reference.
- **Anchor 2, two seeds.** The fine-tune on TOFU facts about invented authors, in five wordings, 1250 steps. It is the example of content the model lacks.
- **RL and its shuffled-reward null, two seeds each.** The null shows what the same loop moves without a meaningful reward.
- **Own-samples SFT, one seed.** The same prompts and rate as RL, with the model's own correct samples as targets.
- **u1 and u2.** Both are fine-tunes on GSM8K's human solutions. u1 is the first one, with no warm-up, and u2 has a 10-step warm-up.

**The second stage.** Anchor 2 had no saved state near the reference. Its nearest in norm were 0.60 of it at seed 42's step 20, and 4.6 times at seed 142's step 100. So both seeds were retrained on an L40 and saved at steps 27 to 30. We then read its three answer sets there with the first run's grader.

**New-word recall** is the share of the reference answer's words, absent from the question, that return in the model's answer. 0 means none and 1 means all. The unchanged model reads 0.179 on the trained wording, because some of those words are common. **The filed learning line** is 0.585, halfway from that to the full update.

**The third stage, Amendment 1.** The matched states hold no facts. So the user asked for anchor 2 saved where it does, at its smallest saved state past the learning line, at both seeds. Each seed was retrained on one L40 with the same recipe. From step 500, every 50 steps, the trained-wording answers were read. The first state at or over the line was saved, its other answer sets were read, and the seed stopped there. Seed 42's read at step 500 doubles as a control, because the first TOFU run read the same step on an H100.

## What we expected, and why

- **Do-nothing:** the zero state, step 1 at rate 0, reads 0 and 0.
- **Anchor 1's second seed:** nearest in both at step 100, from its training record.
- **Anchor 2 on the saved list:** nothing close at either seed, which is why the retrain was prepared.
- **The GSM8K arms:** nearest in norm at step 100 at both seeds and not reaching it, at 0.27 to 0.37 from their training records.
- **KL leans, no record to set them from:** RL, the null and own-samples SFT stay under the reference through step 100. u1 and u2 reach it before step 10. Anchor 2 crosses it between steps 50 and 500.
- **The retrain:** norm-nearest about step 30, KL-nearest between 25 and 45.
- **Learning:** not learned at any matched state, with trained-wording recall under 0.30. Step 30 sits inside a 250-step warm-up, so little fact learning was expected.
- **The first state past the line:** step 550 or 600 at both seeds, because seed 42 sat 0.015 under the line at step 500 and was rising. The seeds' steps within 100 of each other. That puts its norm at 31 to 34 times anchor 1's. Rephrased recall over 0.40 and under the full update's 0.53.
- **The card control:** the L40 rerun of seed 42's step 500 within 0.03 of the H100's 0.570, the spread across cards being unknown. A miss would be reported, not a stop.

## What we observed

### Does the table exist, and which arms reach the reference?

Figure: `experiments/qwen_size_grid/figures/norm_vs_kl.png`.

**Reading it.** Yes: every arm has a named state, but only some reach the reference. Anchor 2 passes almost through it, and u1 and u2 pass above it. All five GSM8K runs of RL, the null and own-samples SFT stop to its left, because none reaches the norm by step 100. In KL they differ widely. RL ends just under the reference at both seeds, at 0.88 and 0.91 of it. The null's seeds read 1.13 and 0.45, so they differ by 2.5 times and one of them reaches it. Own-samples SFT stops at 0.011 of the reference, at 0.56 of the norm.

| arm | full update: step, norm, KL | nearest norm: step, ratio, its KL ratio | nearest KL: step, ratio, its norm ratio | reached norm, KL |
|---|---|---|---|---|
| anchor 1, seed 0 | 100, 0.487, 0.0959 | 100, 1.000, 1.000 | 100, 1.000, 1.000 | yes, yes |
| anchor 1, seed 1 | 100, 0.489, 0.0996 | 100, 1.005, 1.039 | 100, 1.039, 1.005 | yes, yes |
| anchor 2, seed 42 | 1250, 18.109, 0.5029 | 29, 0.981, 1.068 | 27, 0.983, 0.900 | yes, yes |
| anchor 2, seed 142 | 1250, 18.121, 0.5052 | 30, 1.019, 1.133 | 27, 0.991, 0.895 | yes, yes |
| RL, seed 0 | 100, 0.351, 0.0847 | 100, 0.721, 0.883 | 100, 0.883, 0.721 | no, no |
| RL, seed 70 | 100, 0.347, 0.0872 | 100, 0.713, 0.909 | 100, 0.909, 0.713 | no, no |
| null, seed 0 | 100, 0.370, 0.1087 | 100, 0.760, 1.133 | 100, 1.133, 0.760 | no, yes |
| null, seed 70 | 100, 0.335, 0.0429 | 100, 0.689, 0.447 | 100, 0.447, 0.689 | no, no |
| own-samples SFT, seed 0 | 100, 0.274, 0.0010 | 100, 0.563, 0.011 | 100, 0.011, 0.563 | no, no |
| u1 | 100, 3.378, 0.6624 | 1, 0.772, 1.105 | 1, 1.105, 0.772 | yes, yes |
| u2 | 100, 2.858, 0.5201 | 7, 0.970, 2.059 | 5, 1.192, 0.549 | yes, yes |

Ratios are to the reference. For the GSM8K arms these nearest states are flagged last states, not matches, except the null at seed 0 in KL. From `results/size_grid.json`.

**The two sizes agree closely for anchor 2 and not for u2.** Anchor 2's two nearest states sit two or three steps apart at each seed. At its norm-nearest step, u2 has moved the maths rows about twice as far as the reference. u1's nearest state in both is its first step, the smallest non-zero state it has.

**Own-samples SFT moves the maths rows far less at a similar norm.** At step 100 RL's KL is about 80 times SFT's, at 83 and 86 times. SFT's norm there is 0.78 and 0.79 of RL's, so the norms are similar, not equal. RL's step 50, whose norm is below SFT's step 100, still reads 56 and 66 times SFT's KL. We do not name a cause.

**The earlier read on sampled answers agrees.** The E69 report's KL, read on rollouts in `qwen_rl/results/lr1e-6w10/{R,N,F}.json` under `.reads`, gives the same order at step 100.

| KL at step 100 | RL, first seed | RL, second seed | null, first seed | null, second seed | own-samples SFT |
|---|---|---|---|---|---|
| first 16 maths rows, this report | 0.0847 | 0.0872 | 0.1087 | 0.0429 | 0.0010 |
| rollouts, E69 | 0.194 | 0.196 | 0.321 | 0.137 | 0.0017 |

On both reads the null sits above RL at the first seed and below it at the second, with own-samples SFT about two orders of magnitude under RL. The second seed's rollout numbers are in `lr1e-6w10_s70/{R,N}.json`.

### Has anchor 2 learned its facts at that size?

Figure: `experiments/qwen_size_grid/figures/recall_vs_norm.png`.

**Reading it.** No. The filed rule reads "not learned" at all four matched states, both seeds, as expected. Trained-wording recall reads 0.157 to 0.173, against 0.179 for the unchanged model and 0.99 for the full update at both seeds. That is a little under the unchanged model, and we do not name a cause. The trained wording is the easiest test, so this is the strong form of the statement.

| seed, step | norm | KL | trained | rephrased | floor | key trained | key rephrased | label |
|---|---|---|---|---|---|---|---|---|
| 42, 27 | 0.438 | 0.0943 | 0.159 | 0.163 | 0.152 | 0.055 | 0.082 | not learned |
| 42, 29 | 0.477 | 0.1025 | 0.157 | 0.158 | 0.152 | 0.058 | 0.084 | not learned |
| 142, 27 | 0.436 | 0.0951 | 0.172 | 0.171 | 0.162 | 0.060 | 0.082 | not learned |
| 142, 30 | 0.496 | 0.1087 | 0.173 | 0.173 | 0.169 | 0.062 | 0.084 | not learned |
| unchanged model | 0 | 0 | 0.179 | 0.166 | 0.190 | 0.071 | 0.090 | cannot pass |
| full, seed 42, step 1250 | 18.109 | 0.5029 | 0.993 | 0.532 | 0.211 | 0.462 | 0.404 | learned |
| full, seed 142, step 1250 | 18.121 | 0.5052 | 0.992 | 0.541 | 0.210 | 0.462 | 0.392 | learned |

The floor is recall on 400 rows about authors never trained. It also reads under the unchanged model at every matched state. Rephrased recall straddles the unchanged model's. From `results/matched.json`; unchanged and full from `qwen_tofu_forms/results/read.json` and `qwen_anchor_checks/results/S.json`.

So at this size a matched-step reading compares the template fine-tune, finished, with the first steps of a fine-tune, not with a model that has new facts.

### How large is the smallest saved state that holds its facts?

Same figure, whose inset enlarges the region around the line. **Reading it.** About 30 times anchor 1's norm, at both seeds. Seed 42's first state past the line is step 550, and seed 142's is step 500. Trained-wording recall there reads 0.615 and 0.591, over the line of 0.585 and far under the full update's 0.99. Rephrased recall reads 0.434 and 0.439, against 0.166 for the unchanged model and 0.53 and 0.54 for the full update, so the facts carry partly to new wordings. These two states have 31.9 and 30.2 times anchor 1's norm. On the maths rows, which they did not train on, their KL is 3.47 and 3.45 times anchor 1's.

| seed, state, card | norm | times anchor 1's norm | trained-wording recall | against the line of 0.585 |
|---|---|---|---|---|
| 42, step 200, H100 | 5.597 | 11.5 | 0.262 | under |
| 42, step 500, H100 | 14.667 | 30.1 | 0.570 | under by 0.015 |
| 42, step 500, L40 | 14.682 | 30.2 | 0.583 | under by 0.002 |
| **42, step 550, L40** | 15.510 | 31.9 | 0.615 | **first past it** |
| **142, step 500, L40** | 14.676 | 30.2 | 0.591 | **first past it**, over by 0.006 |
| 42, step 1000, H100 | 18.068 | 37.1 | 0.978 | over |
| 42, step 1250, H100 | 18.109 | 37.2 | 0.993 | over |

| state | KL | times anchor 1's KL | rephrased | floor | key trained | key rephrased |
|---|---|---|---|---|---|---|
| 42, step 550 | 0.3326 | 3.47 | 0.434 | 0.209 | 0.405 | 0.348 |
| 142, step 500 | 0.3308 | 3.45 | 0.439 | 0.224 | 0.386 | 0.354 |

From `results/past_line.json`, and for the H100 rows `qwen_tofu_forms/results/read.json`. The floor and key columns are those of the matched-state table above.

**Each size is an upper bound on the crossing's.** The grid reads every 50 steps from step 500. At seed 42 on the L40 the crossing therefore lies between steps 500 and 550, at 30.2 to 31.9 times anchor 1's norm. At seed 142 the first read was already over the line, and its last read under it is step 30, so its crossing is not bracketed from below.

**A caution: "first past the line" is not sharp between steps 500 and 550.** The two cards read seed 42's step 500 at 0.583 and 0.570, which is 0.013 apart. On the L40, seed 42's step 500 fell 0.002 under the line and seed 142's fell 0.006 over it. Both margins are smaller than that card difference, so another card could have named a different step at either seed. The size barely moves if it did, since the two steps differ by under two times anchor 1's norm.

**What this leaves true about size.** Anchor 1 clears its own filed learning rule at the reference size: its format rate reads 0.926 and 0.934 at its two seeds, against a line of 0.74. That rule counts answer format, not new words, so the two lines are not one scale. Even so, the smallest saved state of anchor 2 holding its facts is about 30 times anchor 1's norm. So the two anchors cannot be compared at equal size while both have done their job. This is what was measured, and we do not name a cause.

## What we learned

Size matching and learning do not line up across the two examples. The state that matches anchor 1's size is available and on the full update's path, but it reads like the unchanged model on the facts it is meant to teach. Any reading through the decomposition at that state therefore says what the opening steps of a facts fine-tune do, not what new content looks like. Even the smallest saved state holding the facts is about 30 times anchor 1's norm.

The plan named the matched step as the one the comparison leans on, and the user's answer replaces that: read both, with the matched step a size control only. Any claim that an update adds something new rests on the full update, with its size stated. Anchor 2's first states past the line are saved at both seeds for the rows to read. The user called this a weak endorsement, not having fully followed the question, and `decisions.md` records that. For the GSM8K arms, the table says their last states sit under the reference's norm, so their full update and their nearest state are the same state.

## How compelling is the evidence

- **What the norm is blind to, as filed:** where the update lands.
- **What KL here is blind to, as filed:** it is on maths rows only, so it is off-task for the TOFU arm and says nothing of what it learned on facts. That matters for anchor 2's KL-nearest step, which matches the reference on rows the update does not train on.
- **What neither says, as filed:** that a matched state has learned its task. That is read from the task-row answers.
- **What recall is blind to, as filed:** it counts new words only, so a state that has shifted toward fact-like answers without the facts reads near the unchanged model. Batch loss has already fallen from 6.24 to 2.39 by seed 42's step 30, with recall flat. We do not name a cause.
- **What "past the line" is blind to, as filed:** the trained wording is the easiest test, so it says the facts are held as trained, not that they carry to rephrasings. The rephrased read above says that separately.
- **Controls.** The zero state read 0 and 0 exactly. Anchor 1's step 100 read the earlier run's numbers within the band, so the job went on.
- **Card control.** The L40 rerun of seed 42's step 500 reads 0.583 against the H100's 0.570, 0.013 apart, inside the filed band of 0.03. That is one state on one pair of cards, so it is one measurement of the spread.
- **Same path.** The retrain's step 100 against the saved file reads 0 at seed 142, where both ran on an L40. It reads 0.74% of the norm at seed 42, an H100 file against an L40 rerun. The two seeds' files sit 90% of the norm apart, and the filed bar was a tenth of that, so the matched states lie on the full update's path.
- **Seeds.** Two for each anchor, RL and null, and one for own-samples SFT. Recall along training has one seed, 42. Seed 142 has answers only at steps 27, 30, 500 and 1250. The null's two seeds differ by 2.5 times in KL, so its row is not one number.
- **Precision.** u1 and u2 exist only as bf16 files, and all the other arms as fp32.
- **Noise.** Recall at the matched states differs by 0.016 between the seeds. The gap to the unchanged model is 0.006 to 0.022, so the two are of similar size.

## Surprises

Two filed leans missed, both leans and not gates.

- **The null at the first seed reached the KL reference.** We leaned that it would stay under through step 100, and it reads 1.13 of it at step 100. At the second seed it reads 0.45, as leaned.
- **Anchor 2 crossed the KL reference earlier than leaned.** We leaned between steps 50 and 500. The saved list brackets it between steps 20 and 50, and the retrain puts it between steps 27 and 28 at both seeds.

Two filed expected values missed in the third stage, both at seed 142 and both from one event. It was already over the line at step 500, where we expected step 550 or 600. So its size reads 30.2 times anchor 1's norm, where we expected 31 to 34.

One error in the first read. `run.py`'s first list left out saved states of u1, step 1, and of u2, steps 1, 7, 8 and 9. The lead found this reading the first output. They were added and read before the grid was committed, and `predictions.json` stands unedited. It changed both arms' norm-nearest step: u1 from 2 to 1, and u2 from 10 to 7. It also moved u1's KL-nearest step from 2 to 1.

## What next

1. **Reads through the decomposition at these states.** They are `qwen-decomp-lead`'s, at the matched step, the first state past the line and the full update. If the two anchors still separate at the matched step, that separation cannot come from facts the update recalls.
2. **Answers at seed 142 under step 500,** only if a row's reading turns on where exactly that seed crosses the line. Nothing filed does today.

## Departures from the literature

None: this report cites no outside source for a definition, recipe or threshold. The recipes are the project's own earlier runs, unchanged.

## Scorecard

| Filed item | Rule or claim | Read | Verdict |
|---|---|---|---|
| zero-state control | anchor 1 step 1 reads 0, 0; a miss stops the job | 0, 0 | passes |
| E78 control | anchor 1 step 100 within 1e-4 of 0.48652034 and 0.09593200 | 0.48652034, 0.09593199 | passes |
| `do_nothing` | zero state is the reference for nothing | 0, 0 | held |
| `anchor1_s1` | step 100, norm 0.4887, KL 0.0996 | 0.4887, 0.0996 | held |
| `anchor2_s0_norm` | step 20 at 0.292, ratio 0.60, on the saved list | as filed | held; replaced by the retrain's steps 27 and 29 |
| `anchor2_s142_norm` | step 100 at 2.217, ratio 4.6, on the saved list | as filed | held; replaced by the retrain's steps 27 and 30 |
| `gsm8k_arms_norm` | step 100, not reached, 0.27 to 0.37 | 0.274 to 0.370, none reached | held |
| `kl_leans`, GSM8K | RL, null, SFT under the reference through step 100 | null at seed 0 reads 1.13 | **lean missed** on that one run |
| `kl_leans`, u1 and u2 | reach it before step 10 | u1 at step 1, u2 by step 5 | held |
| `kl_leans`, anchor 2 | crosses between steps 50 and 500 | between 20 and 50 | **lean missed** |
| `t_norm` | seed 42 about 30, seed 142 within two steps | 29, 30 | held |
| `t_kl` | between 25 and 45 | 27, 27 | held |
| `learned` | not learned, trained-wording recall under 0.30 | not learned, 0.157 to 0.173 | held |
| rerun control | rerun distance under a tenth of the seeds' distance | 0.0074 and 0, against 0.899 | on the full update's path |
| Amendment 1 control | L40 rerun of seed 42 step 500 within 0.03 of the H100's 0.570 | 0.583, 0.013 apart | passes |
| Amendment 1 `step` | step 550 or 600 at both seeds | 550 at seed 42, 500 at seed 142 | **missed** at seed 142 |
| Amendment 1 `size` | about 31 to 34 times anchor 1's norm | 31.9 at seed 42, 30.2 at seed 142 | **missed** at seed 142 |
| Amendment 1 `para` | rephrased recall over 0.40, under 0.53 | 0.434, 0.439 | held |
| Amendment 1 `seeds` | the two seeds' steps within 100 | 50 apart | held |

## Provenance

Cluster L40s, `debug` QoS, $0. All under `experiments/qwen_size_grid/`. Job 52219 read the grid, 1051 s by `results/run.log`. Job 52238 read the five missed states, 81 s, then retrained both seeds and read the answers, 1072 s, writing `results/matched.json`. Job 52310 ran Amendment 1 on an L40, `debug` QoS, $0, 1324 s by `results/run.log`, writing `results/past_line.json`.

- Reads: `results/reads.jsonl`, one line per state. Grid: `results/size_grid.json`, which names each anchor 2 seed's `past_line` state.
- Saved states: `results/large/dW/a2_s42/step_{27,29,550}.safetensors` and `a2_s142/step_{27,30,500}.safetensors`. Answers: `results/large/answers/`.
- Recall along training: `experiments/qwen_tofu_forms/results/read.json`, its norm by step in `train.json`. Second seed's full update: `experiments/qwen_anchor_checks/results/S.json`.
- Code: `run.py`, `matched.py`, `past_line.py`. Figures: `plot.py`, `figures/`.

The filed learning line of 0.585 was set from 0.179 and 0.992. The full update's recall reads 0.9927 at seed 42 and 0.9917 at seed 142, and no state's label depends on the difference.

## Not run

- Think-mode answer sets at any state.
- Reads through the decomposition, which are `qwen-decomp-lead`'s.
- Bare and real-fact answers at the matched states and the states past the line.
- u1's and u2's answers at their matched steps.
- A second seed of own-samples SFT.
- Answers at seed 142 between steps 30 and 500.
- Answers on the L40 reruns after each seed's first state past the line.
- Rephrased answers on the L40 at seed 42's step 500.
