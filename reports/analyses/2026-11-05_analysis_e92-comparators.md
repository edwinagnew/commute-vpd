# What do three cheap measures say about the Qwen updates, on the same weights as the decomposition's rows? (E92)

We read three cheap measures on the saved Qwen updates at matched and full size: update size, the unchanged model's rank of the trained answer, and one steering vector. One vector recovers most of the facts update's gain at the template update's size and a small part at full size, and at matched size that update gained on tokens the unchanged model rated very low without putting more right tokens first. For `GOALS.md` these columns are comparators only, because a one-number steering share reads differently at different steps of one update and cannot say whether an update is new or reuses what the model has.

*2026-11-05, `baseline-lead`. Plan `reports/plans/2026-11-05_plan_e92-comparators.md`, rules `experiments/qwen_comparators/results/predictions.json` (unedited), notebook `notebooks/E92_qwen_comparators.ipynb`, figures `experiments/qwen_comparators/figures/`, from `plot.py`, after `dissect.py`.*

**Status: provisional, no independent check.** Every number is copied from `results/states/*.json`, `W0_*.json`, `controls.json`, or from `results/dissection.json` and `results/comparator_table.json`, which `dissect.py` writes from the saved per-position files on CPU. Those two files and everything they alone support were read after the run and were not filed.

**Step 550 at seed 42 and step 500 at seed 142 are an unfiled addition.** They were not in the plan. Each is that seed's first saved state past E91's learning line. They are labelled wherever they appear.

**Names.** "Template update" is anchor 1, `anchor1_s0` and `anchor1_s1`. "Facts update" is anchor 2, the TOFU fine-tune; `anchor2_s0` is seed 42 and `anchor2_s142` is seed 142.

## The answer

No cheap number here separates the template update from the facts update in a way we can trust. One steering vector recovers 0.83 of the facts update's gain at the template's size and 0.065 at its own full size, on both seeds, so the comparison changes direction with the step. At matched size the facts update gains on tokens the unchanged model rated very low, and loses on those it ranked first, without ranking more answers first. The steering share is a comparator, not the definition of new against reuse.

## What we were trying to test

The plan needs a cheap non-decomposition reading beside every D3 row, on the same state, seed and step. A row whose separation a comparator matches says less than its number suggests. Three measures are read, cheapest first, and none reads a decomposition.

- **Update size.** `norm` is the root of the summed squares over the 196 decomposed matrices, `ratio_norm` is that over the unchanged model's norm, 1310.517, and `share_squared` is its square. Do-nothing reads 0. Blind to direction.
- **Rank.** The unchanged model's rank of each next answer token in the whole vocabulary, 1 being the top, with its log P. Two summaries: the answer's first token, and the teacher-forced mean over every answer token, end token included. Blind on the facts rows to anything but trained wording.
- **One steering vector.** The vector is the mean over one half of the rows of the state's block input minus the unchanged model's, at answer positions. It is added to the unchanged model at every answer position on the other half. The share is the rise in log P over the state's gain. The block is the one with the largest first-half share.

**Controls.** The zero update must read as the unchanged model, and ran first and alone: largest log P difference 0.0, ranks equal, no block chosen. The template's update must raise its own tokens: gain 0.666 ± 0.051, defined and positive. The norm must equal E91's stored read within 1e-9: it does on every state E91 read. A random vector at the vector's norm, a Gaussian mix of the unchanged model's centred inputs, asks whether any vector of that size raises log P.

**Rows.** The template update: the prompted model's greedy answers on held-out problems, 64 x 256. The facts update: its trained rows in TOFU's own wording, 128 x 256. The GSM8K arms: gold solutions, 103 x 256, because the D3 maths rows hold the unchanged model's own answers. These are different rows per arm, so a difference between anchors can come from the rows as well as from the update.

**The amendment's stop rule.** A column counts only if the gap between the anchors' means exceeds the larger same-anchor seed spread.

## What we expected, and why

The leans are in the scorecard. In words: the template's vector recovers a modest part and the random vector none. The facts update at full size gives under 0.05, because one vector cannot hold 787 facts, and ends above log P -0.2 on rows it trained. At matched size it gains under half a nat with a share undefined or under 0.3, because E91 reads no facts there. The GSM8K arms gain under 0.1 on gold tokens. At matched size only the first token's rank counts.

## What we observed

### Is the first-token column a bug in the rows? No, it is real, and no GPU read was needed

![](../experiments/qwen_comparators/figures/first_token_rank.png)

The runner's flag: mean log P of the first answer token reads -15.68 at median rank 21 on template rows and -13.59 at median rank 11 on gold rows. On TOFU rows it reads -7.26 at median rank 1.

**Checks that it is not the rows.**

- `dissect.py` rebuilds all three row sets with `run.py`'s own builders. They match the saved positions, masks and first-token marks exactly, by assertion.
- Decoded, the five tokens before a first token are `assistant\n<think>\n\n</think>\n\n`, from `qwen_eval.prompt` and `qwen_anchors.chat` with `enable_thinking=False`, and the marked target is the answer's opening: "Given: 8 books", "It takes 2/", "Yes, Jaime Vasquez".
- `Rows.first` marks index `p`, the first answer token, and `_reads` moves it one place back, so the position reads that token. `token_reads` then takes the next token at that position.
- The positive control is the template update itself: at those same positions "Given" goes from median rank 21 at log P -15.68 to rank 1 at -0.93, a gain of 14.74 ± 0.69. A misplaced mask could not do that.
- On gold rows "Let", the commonest opening in the unchanged model's own greedy answers, reads rank 1 at log P -0.32. In `maths_rows.json` its openings are "Let" 110 and "Step" 73 of 208.
- The low values survive on answers whose whole question is in the row: template rows 37 first tokens at median rank 25, log P -17.10; gold 69 at rank 8, -13.43.

**Why the numbers look odd.** The unchanged model's opening is sharply peaked. A first token it ranks first reads near 0, and the next ones fall faster than other answer tokens do.

| mean log P by the token's rank | 1 | 2 | 3-5 | 6-10 | 11-30 | 31-100 |
|---|---|---|---|---|---|---|
| Template, first tokens | none | -15.0, one token | -8.9, one token | -10.9 | -15.1 | -18.5 |
| Template, all tokens | -0.1 | -2.8 | -4.9 | -8.0 | -10.9 | -14.4 |
| Gold, first tokens | -0.1 | -2.5 | -7.6 | -13.5 | -15.9 | -19.4 |
| Gold, all tokens | -0.1 | -3.9 | -6.5 | -9.1 | -12.9 | -15.5 |
| TOFU, first tokens | -0.0 | -7.3 | -11.2 | -17.5 | -18.4 | -24.0 |
| TOFU, all tokens | -0.1 | -4.3 | -6.6 | -8.8 | -10.8 | -12.9 |

What the top token's own probability is at each position is not saved, so "peaked" rests on the log P of the tokens below it and on the first-ranked tokens' reading near 0. Logits are not saved.

**Limits on the column.** All 62 template first tokens are one token, "Given". Gold has 51 distinct tokens among 121, TOFU 98 among 464, and 256 of TOFU's 464 are ranked first already. So the first-token rank's gap of 20 between the anchors describes one word against a mix of openings. The mean is bimodal on TOFU: the median first-token log P is -0.03 against a mean of -7.26.

### Does the steering share belong to the update, or to the step it is read at?

![](../experiments/qwen_comparators/figures/share_vs_size.png)

**To the step, on this evidence.** The share falls with step on both seeds.

| state | norm | gain on scoring half | share ± se | nats added | random share | nats added, random | vector norm, share of input | variance share, vector, random |
|---|---|---|---|---|---|---|---|---|
| template, seed 0, step 100 | 0.487 | 0.586 | 0.185 ± 0.019 | 0.109 | -0.038 | -0.022 | 164, 0.052 | 0.016, 0.155 |
| template, seed 1, step 100 | 0.489 | 0.588 | 0.182 ± 0.019 | 0.107 | -0.037 | -0.022 | 163, 0.052 | 0.015, 0.155 |
| facts, seed 42, step 29 | 0.477 | 2.095 | 0.827 ± 0.025 | 1.732 | 0.106 | 0.222 | 580, 0.181 | 0.076, 0.013 |
| facts, seed 142, step 30 | 0.496 | 2.106 | 0.829 ± 0.025 | 1.745 | 0.108 | 0.227 | 589, 0.184 | 0.077, 0.013 |
| facts, seed 42, step 550, *unfiled addition* | 15.510 | 3.575 | 0.227 ± 0.007 | 0.813 | 0.046 | 0.165 | 469, 0.147 | 0.007, 0.013 |
| facts, seed 142, step 500, *unfiled addition* | 14.676 | 3.536 | 0.238 ± 0.008 | 0.843 | 0.046 | 0.163 | 466, 0.146 | 0.005, 0.013 |
| facts, seed 42, step 1250 | 18.109 | 3.809 | 0.065 ± 0.003 | 0.246 | 0.048 | 0.183 | 504, 0.157 | 0.078, 0.013 |
| facts, seed 142, step 1250 | 18.121 | 3.808 | 0.067 ± 0.003 | 0.256 | 0.046 | 0.175 | 490, 0.153 | 0.073, 0.013 |

- **The numerator falls too.** The vector's added nats fall from 1.73 to 0.25, while its own norm stays between 466 and 589. So the fall is not only a larger gain below it.
- **At the full update the vector barely beats random in nats:** 0.25 against 0.18. The filed test still reads it above random on every state.
- **The block is 27, the last, on every anchor state.** At step 29 every block from 20 up reads 0.69 or more on the first half; at step 1250 no block reads over 0.061.

**The stop rule.**

| where | column | anchor 1, two seeds | anchor 2, two seeds | gap | larger spread | verdict |
|---|---|---|---|---|---|---|
| full update | size, `share_squared` | 1.378e-07, 1.391e-07 | 1.909e-04, 1.912e-04 | 1.91e-04 | 2.69e-07 | counts |
| full update | first token's rank, unchanged model | 21, 21 | 1, 1 | 20 | 0 | counts, trivially |
| full update | mean log P, unchanged model | -1.538, -1.538 | -4.248, -4.248 | 2.71 | 0 | counts, trivially |
| full update | gain, all tokens | 0.6661, 0.6657 | 3.768, 3.764 | 3.10 | 0.0041 | counts |
| full update | steering share | 0.1852, 0.1821 | 0.0645, 0.0671 | 0.118 | 0.0031 | counts |
| matched size | size, `share_squared` | 1.378e-07, 1.391e-07 | 1.326e-07, 1.432e-07 | 5.6e-10 | 1.05e-08 | does not count, by construction |
| matched size | first token's rank, unchanged model | 21, 21 | 1, 1 | 20 | 0 | counts, trivially |
| matched size | mean log P, unchanged model | -1.538, -1.538 | -4.248, -4.248 | 2.71 | 0 | counts, trivially |
| matched size | gain, all tokens | 0.6661, 0.6657 | 2.052, 2.062 | 1.39 | 0.0098 | counts |
| matched size | steering share | 0.1852, 0.1821 | 0.827, 0.829 | 0.644 | 0.0031 | counts, reversed |
| past the line, *unfiled addition* | size, `share_squared` | 1.378e-07, 1.391e-07 | 1.401e-04, 1.254e-04 | 1.33e-04 | 1.47e-05 | counts |
| past the line, *unfiled addition* | gain, all tokens | 0.6661, 0.6657 | 3.522, 3.483 | 2.84 | 0.039 | counts |
| past the line, *unfiled addition* | steering share | 0.1852, 0.1821 | 0.227, 0.238 | 0.049 | 0.011 | counts |

**Which columns count.** Update size at full size only. Gain at both. Steering share at both, in opposite directions. The first token's rank and the unchanged model's log P pass trivially at every step: they do not depend on the update, have no seed spread, and describe the rows.

**What this says about a one-number share across anchors.** Facts reads 0.64 above the template at matched size and 0.12 below it at full size. Both clear the rule by far, so the rule alone does not make the column readable. Between steps 550 and 1250 the facts update's share passes the template's 0.185 on both seeds, so at some unsaved step the column would not separate them at all; that is inferred from the two ends, not read.

**Reading anchor 1 against anchor 2.** The template's 0.185 is at norm 0.487, which is its full update and the reference for "matched". It is a single point: its smaller saved steps, 10, 20, 50, are unread, so whether its share also falls with step is not known. Equal norm is not equal effect on these rows. At equal norm the facts update's vector is about 3.5 times the template's and its gain 3 times larger. The unchanged model also starts lower on facts rows, -4.25 against -1.54. As a share of that shortfall the two gains are close, 0.43 and 0.48, and 0.89 at the full update; that ratio is post hoc.

**The share does not fall with step as a rule.** From step 10 to step 100: RL first seed 0.645 to 0.296, second seed 0.688 to 0.914, null first seed 0.541 to 0.729, null second seed 0.552 to 0.552, own-samples 0.386 to 0.495. The gain changes sign on two of those, RL's second seed and the null's first, so those shares are of different things.

### What has the facts update gained at the template's size?

![](../experiments/qwen_comparators/figures/where_gain_lands.png)

![](../experiments/qwen_comparators/figures/rank_one_share.png)

E91 reads trained-wording recall of new words there as 0.157 and 0.173, against 0.179 unchanged, yet the state gains about 2 nats a token. Recall counts new words only, and its filed blind line says a shift without the facts reads near the unchanged model.

**What the files show.**

| per position | unchanged, facts rows | seed 42, step 29 | seed 142, step 30 | seed 42, step 550, *unfiled addition* | seed 42, step 1250 | template rows: unchanged to seed 0, step 100 |
|---|---|---|---|---|---|---|
| share ranked first | 0.549 | 0.560 | 0.562 | 0.834 | 0.921 | 0.775 to 0.828 |
| same, whole question in row | 0.587 | 0.599 | 0.601 | 0.877 | 0.970 | 0.846 to 0.958 |
| first tokens ranked first | 256 of 464 | 217 | 220 | 303 | 356 | 0 to 45 of 62 |
| median first-token rank | 1 | 2 | 2 | 1 | 1 | 21 to 1 |
| median log P, all tokens | -0.347 | -0.880 | -0.884 | -0.028 | -0.001 | -0.001 to -0.000 |
| gain, tokens ranked first by the unchanged model | | -0.28 | -0.29 | -0.07 | 0.02 | -0.01 |
| gain, ranks 2-5 | | 3.19 | 3.19 | 4.81 | 5.11 | 1.78 |
| gain, ranks 6-50 | | 5.64 | 5.67 | 9.01 | 9.53 | 5.46 |
| gain, below 50 | | 7.21 | 7.29 | 12.44 | 13.03 | 3.86 |
| positions that lose, and their mean | | 0.581, -0.32 | 0.581, -0.32 | 0.405, -0.20 | 0.243, -0.22 | 0.255, -0.13 |
| gain carried by the worst three deciles of unchanged log P | | 1.026 | 1.028 | 0.916 | 0.899 | 1.012 |

- Tokens above log P -0.1 fall from 0.459 to 0.281 and 0.277. Tokens under -10 fall from 0.169 to 0.026 and 0.025.
- First tokens' median rank worsens from 1 to 2 while their mean log P rises 4.55 and 4.52 nats.
- Per-position gain correlates 0.9994 between the two seeds, -0.918 with the unchanged log P, and 0.934 with the gain at step 1250.
- By token class at step 29: punctuation -3.67 to -0.96, capitalised -5.49 to -3.05, lower-case word starts -4.86 to -2.53. Of the 300 largest gains, 63 are full stops. By step 1250 these read -0.23, -0.69 and -0.44.
- E84's training record agrees on the gain: batch loss 6.24 at step 1, 2.66 at step 29, 2.39 at step 30. The drop was on file before the lean was filed; the lean read "not learned" from recall.

**What the vector shows.** It is a constant added to block 27's input at 18% of that input's root mean square, against 5% for the template. It recovers 0.827 and 0.829 of the gain, and 0.807 and 0.822 on first tokens. Its variance share among the unchanged model's inputs is 0.076 against 0.013 for the random vector, which is unmatched. The random vector at that norm adds 0.22 nats. Added nats rise with norm across the six facts states' random vectors, 0.163 at 466 to 0.227 at 589; six points on one row set.

**What we infer and have not read.** This is the shape of a flatter output distribution: less sure where it was right, much less wrong where it was wrong, and little change in order. A shift toward TOFU's answer style would also fit, given the punctuation. Neither is read: logits and entropy are not saved, and steered per-position reads are not saved, so nothing here shows which tokens the vector raises or that it flattens anything. We name no cause. That a random vector raises log P on these rows and lowers it on template rows fits a disturbance earning log P where the unchanged model is confidently wrong, which is an inference too.

**What the steering share is blind to.** One mean vector cannot express an input-dependent change, so a low share does not say which mechanism is new. A high share does not say the mechanism is reused: it is a share of mean log P, and this state raised mean log P without ranking more tokens first. It adds at every answer position, including after earlier steered ones. It is a comparator, not the definition of new against reuse: that definition, in `decisions.md`, names a capability and removes it, and this reads neither.

### The apparatus finding: many answers lack their question

Rows are packed and cut, as the maths rows are, so a row may open inside a prompt or an answer. `dissect.py` splits positions by what the row holds of the question.

| rows | whole question | question cut | no question, a fragment |
|---|---|---|---|
| template, positions | 2,414, 27% | 3,408, 38% | 3,244, 36% |
| TOFU, positions | 13,421, 81% | 1,699, 10% | 1,426, 9% |
| gold, positions | 4,053, 34% | 4,875, 41% | 2,873, 24% |
| template seed 0, step 100: log P | -1.41 to -0.15 | -1.71 to -0.87 | -1.45 to -1.40 |
| facts seed 42, step 29: log P | -4.08 to -1.85 | -5.19 to -3.01 | -4.69 to -4.52 |
| facts seed 42, step 1250: log P | -4.08 to -0.12 | -5.19 to -1.17 | -4.69 to -3.03 |
| facts seed 142, step 1250: log P | -4.08 to -0.12 | -5.19 to -1.19 | -4.69 to -3.09 |

The steering share, the gain and the unchanged model's column all pool these, in different proportions per arm. The share cannot be split by class from saved files, because steered positions are not saved. This is the filed rule, and it is not a bug in E92's code; E92's rows match E57's and E86's by the plan's check.

### Do the GSM8K arms move gold tokens? Yes, past the filed band

![](../experiments/qwen_comparators/figures/gsm8k_gain.png)

At step 100 RL reads -0.277 and +0.163, the null 0.283 and 0.595, own-samples 0.048 on its one seed. The null gains most, so gold-token gain does not track a meaningful reward. The share ranked first was 0.686 unchanged and reads 0.683, 0.686, 0.675, 0.696 and 0.687. Answers with some of their question in the row carry the change. On whole-question positions the null's second seed reads -3.04 to -2.17 and RL's first seed -3.04 to -3.52, and fragments stay within 0.02. These read as description only. A gold-token gain is not the arm's accuracy, and this is not a claim about it.

## What we learned

For new against existing, these columns are comparators to set beside the decomposition's rows, and none of them decides it. A one-number steering share needs its step and size beside it, because it changes more than twelve-fold within one update on the same rows and reverses against the template. Mean log P gain can rise without more right tokens coming first, so a gain at matched size is not evidence of learning, and the facts update at that size has not shown any. E91's call that the matched step is a size control only stands, and this supports it from a second direction. The decomposition's rows are still to be read.

## How compelling is the evidence

- **Seeds.** Two on each anchor, and they agree closely. Own-samples has one.
- **Rows.** Each arm is on its own rows, so anchor gaps mix row and update. Trained rows for the facts update and held-out problems for the template.
- **The random control** is matched on norm only. Its variance share runs from 0.013 to 0.401 and the vector's from 0.003 to 0.085. It beats by the filed test on all 18 states.
- **The readings of the matched state are per position and agree across seeds,** but the flatter-distribution reading is inferred.

## Surprises

**Something unexpected happened that I cannot explain.** We predicted the facts update's share at matched size would be undefined or under 0.3, because that state had learned nothing, and it read 0.83 on both seeds. We predicted its gain under half a nat, and it read 2.05 and 2.06. The section above says what the files show. Why one constant vector carries that much of it is not explained.

- **Log P at the full update, -0.48 against a filed above -0.2.** The miss stands. The files give its cause: positions with the whole question read -0.12, those with it cut about -1.2, fragments about -3.0. Fragments' end tokens read -14.9 and -15.4, against -0.02 and -0.01 on all other answers.
- **The first token's rank worsens while its log P rises,** at matched size on both seeds.
- **The vector is largest at the smallest update:** 580 and 589 at steps 29 and 30, against 504 and 490 at step 1250.
- **The vector's variance share is not monotone in step:** 0.076, 0.007, 0.078 at seed 42.
- **The template's vector recovers less than random on first tokens:** 0.021 against 0.048. Its share is carried by later tokens; the first token's 15-nat gain is not reached by the vector.
- **All gain-side leans for GSM8K missed,** with row standard errors far smaller than the lean assumed.

## What next

1. **A 2-minute read on one cluster L40, `debug`, $0, not run here; it needs a `pod-runner`.** Save per-position log P and output entropy at the unchanged model, the state, and the unchanged model plus the vector and plus the random vector, at the facts update's steps 29 and 30 and 1250 and the template's step 100, and add a variance-matched random vector. If entropy rises under both state and vector, the matched share is reading the flattening, and it should not go beside a D3 row as evidence on mechanism. It also gives the share on whole-question positions.
2. **The template's share at its saved steps 10, 20 and 50, in the same job.** If it falls with step too, the table compares curves. If it is flat, the fall belongs to the facts update.
3. **Whether to file the share ranked first as a comparator before the D3 rows are read.** It separated the anchors at matched size here, but after the fact, so it counts only once filed.

## Scorecard

Filed leans, against the rule unedited. A miss stands.

| id | claim | rule | read | verdict | cause, where the files allow |
|---|---|---|---|---|---|
| C1 | Zero state reads as the unchanged model | under 1e-5, ranks equal, no block | 0.0, equal, none | held | |
| C2 | Template raises its own tokens | gain defined, positive | 0.666 ± 0.051 | held | |
| C3 | Norm equals E91's | within 1e-9 | equal on every state E91 read; steps 29, 30, 500, 550 not in E91's reads | held | |
| D1 | Do-nothing | norm 0, gain 0, share undefined | as filed | held | |
| T0 | Template size | 0.48652, 3.712e-04, 1.378e-07 | 0.48652, 3.712e-04, 1.378e-07 | held | |
| T1 | Template, first-token rank | above 10, then 1 | 21 to 1 | held | |
| T2 | Template, all-tokens rank | median 1 at the state | 1, was 1 already | held, blind | The median was 1 before the update |
| T3 | Template gain | at least 0.3 | 0.666, 0.666 | held | |
| T4 | Template share | 0.05 to 0.4 | 0.185, 0.182 | held | |
| T5 | Template random | within 2 se of 0 and under 0.05 | -0.038 ± 0.007, -0.037 ± 0.006 | missed on 2 se, held on size | The random vector lowers log P on these rows; its norm matches, its variance share 0.155 does not |
| A0 | Facts full size | 18.1087, 1.3816e-02, 1.909e-04 | 18.1087, 1.3818e-02, 1.909e-04 | held | Filed from ‖W0‖ 1310.75; read 1310.517 |
| A1 | Facts, unchanged log P | -4 to -5.5 | -4.25 | held | |
| A2 | Facts, state's log P | above -0.2 | -0.480, seed 142 -0.484 | missed | The pooled rows include answers with no question; whole-question positions read -0.12 |
| A3 | Facts, unchanged first rank | 1 to 3 | 1 | held | |
| A4 | Facts, full update's share | under 0.05 | 0.0645, seed 142 0.0671 | missed | The random vector alone reads 0.048 and 0.046, so the bar sat at the control's own level |
| M0 | Matched size | about anchor 1's | 0.477, 0.496 against 0.487 | held | |
| M1 | Matched gain | within 0.5 | 2.05, 2.06 | missed | Recall was read as if it were log P; E84's batch loss had already fallen to 2.4 to 2.7 |
| M2 | Matched share | undefined or under 0.3 | 0.827, 0.829 | missed | See Surprises; not explained |
| G0 | GSM8K size | 2.68e-04, 2.82e-04, 2.09e-04 | 2.68e-04, 2.82e-04, 2.09e-04 | held | |
| G1 | GSM8K unchanged log P | -2 to -3 | -2.73 | held | |
| G2 | GSM8K gain | under 0.1 on all three, R within 2 se of N | R -0.277, +0.163; N 0.283, 0.595; F 0.048 | missed | Only F is under 0.1; R is more than 2 se from N on each seed; no cause named |
| G3 | GSM8K share | undefined on at least one of five at step 100 | 0.296, 0.914, 0.729, 0.552, 0.495 | missed | Row standard errors are small: F's gain is 0.048 ± 0.004 |
| S1 | Share and gain count at full size | stop rule | both count | held | |
| S2 | At matched size neither does, except the first-token rank | stop rule | the rank counts; gain, share and the unchanged log P count too | missed | Same as M1 and M2; the unchanged log P does not depend on the step |

**Numbers from the runner, checked.** All match the files. One rounding: the template's random reads -0.0378 at seed 0 and -0.0375 at seed 1, given as -.037. The runner's list named two misses; the files give eight, with T5, A4, M1, M2, G3 and S2 added.

## Departures from the literature

- **Steering.** The vector's definition is carried from this project's `steering.py`, E52 and E54, which adds at one answer position. Here it adds at every answer position, because a first token alone can carry little of the gain on the facts rows. The vector is a difference between two models on one input, not a contrast between two behaviours on one model.
- **Random control.** A Gaussian mix of the model's own inputs, not an isotropic direction, which would read 0 by construction. Matched on norm, not on variance.
- **TOFU, Maini et al. 2024.** Read on its trained rows, packed in rows of 256, by mean log P, and not by TOFU's Truth Ratio or forget metrics. The 13 rows E84 dropped stay as context and mark no answer.
- **GSM8K arms.** Gold solutions, not the D3 maths rows, which hold the unchanged model's own answers.
- **New against reuse, Jain et al. 2024 by way of `decisions.md`.** Not tested here: no capability is named and nothing is removed.

## Provenance

Jobs 52271, 52273, 52284 and 52382, all in `results/run.log`, cluster L40, `debug` QoS, $0. Per-position reads in `results/large/`, 118 KB to 215 KB a state. `dissect.py` and `plot.py` run on CPU after the run and load no model. `anchor2_s0`'s steps 29 and 550 and `anchor2_s142`'s are E91's retrained paths; E91's rerun against saved step 100 reads 0.0074 of its norm at seed 42 and 0.0 at seed 142, against a seed distance of 0.90, so they lie on the full update's path.

## The comparator table for the D3 rows

Same state, seed and step where a row exists. **Every D3 cell is a gap:** no D3 row has been read on any of these files. The rows are `qwen-decomp-lead`'s, due 11-07 to 11-09 on the anchors and 11-10 to 11-11 on the GSM8K arms. Own-samples has no second seed, so it has no row to fill. E75's clean table is a D1 read, not a D3 row, and is not joined here. Machine copy: `experiments/qwen_comparators/results/comparator_table.json`.

| state | size: norm, ratio to W0 | first token, median rank, W0 to state | mean log P, all tokens, W0 to state | gain ± se | steering share ± se, block | random share ± se | route split | δW against null | kept under masks | think overlap |
|---|---|---|---|---|---|---|---|---|---|---|
| anchor1_s0, step 100 | 0.487, 3.71e-04 | 21 to 1 | -1.54 to -0.87 | 0.666 ± 0.051 | 0.185 ± 0.019, 27 | -0.038 ± 0.007 | gap | gap | gap | gap |
| anchor1_s1, step 100 | 0.489, 3.73e-04 | 21 to 1 | -1.54 to -0.87 | 0.666 ± 0.051 | 0.182 ± 0.019, 27 | -0.037 ± 0.006 | gap | gap | gap | gap |
| anchor2_s0, step 29 | 0.477, 3.64e-04 | 1 to 2 | -4.25 to -2.20 | 2.052 ± 0.045 | 0.827 ± 0.025, 27 | 0.106 ± 0.007 | gap | gap | gap | gap |
| anchor2_s142, step 30 | 0.496, 3.78e-04 | 1 to 2 | -4.25 to -2.19 | 2.062 ± 0.046 | 0.829 ± 0.025, 27 | 0.108 ± 0.008 | gap | gap | gap | gap |
| anchor2_s0, step 550, *unfiled addition* | 15.510, 1.18e-02 | 1 to 1 | -4.25 to -0.73 | 3.522 ± 0.064 | 0.227 ± 0.007, 27 | 0.046 ± 0.003 | gap | gap | gap | gap |
| anchor2_s142, step 500, *unfiled addition* | 14.676, 1.12e-02 | 1 to 1 | -4.25 to -0.77 | 3.483 ± 0.061 | 0.238 ± 0.008, 27 | 0.046 ± 0.004 | gap | gap | gap | gap |
| anchor2_s0, step 1250 | 18.109, 1.38e-02 | 1 to 1 | -4.25 to -0.48 | 3.768 ± 0.067 | 0.065 ± 0.003, 27 | 0.048 ± 0.004 | gap | gap | gap | gap |
| anchor2_s142, step 1250 | 18.121, 1.38e-02 | 1 to 1 | -4.25 to -0.48 | 3.764 ± 0.067 | 0.067 ± 0.003, 27 | 0.046 ± 0.003 | gap | gap | gap | gap |
| R_s0, step 10 | 0.069, 5.24e-05 | 11 to 11 | -2.73 to -2.93 | -0.198 ± 0.013 | 0.645 ± 0.042, 23 | -0.042 ± 0.005 | gap | gap | gap | gap |
| R_s0, step 100 | 0.351, 2.68e-04 | 11 to 30 | -2.73 to -3.01 | -0.277 ± 0.018 | 0.296 ± 0.019, 23 | -0.038 ± 0.005 | gap | gap | gap | gap |
| R_s70, step 10 | 0.068, 5.17e-05 | 11 to 7 | -2.73 to -2.85 | -0.118 ± 0.009 | 0.688 ± 0.046, 23 | -0.045 ± 0.006 | gap | gap | gap | gap |
| R_s70, step 100 | 0.347, 2.65e-04 | 11 to 7 | -2.73 to -2.57 | 0.163 ± 0.027 | 0.914 ± 0.066, 27 | 0.176 ± 0.018 | gap | gap | gap | gap |
| N_s0, step 10 | 0.065, 4.95e-05 | 11 to 10 | -2.73 to -2.80 | -0.069 ± 0.004 | 0.541 ± 0.033, 23 | -0.034 ± 0.004 | gap | gap | gap | gap |
| N_s0, step 100 | 0.370, 2.82e-04 | 11 to 20 | -2.73 to -2.45 | 0.283 ± 0.028 | 0.729 ± 0.052, 27 | 0.146 ± 0.014 | gap | gap | gap | gap |
| N_s70, step 10 | 0.065, 4.93e-05 | 11 to 15 | -2.73 to -2.59 | 0.137 ± 0.008 | 0.552 ± 0.035, 23 | 0.035 ± 0.004 | gap | gap | gap | gap |
| N_s70, step 100 | 0.335, 2.56e-04 | 11 to 19 | -2.73 to -2.13 | 0.595 ± 0.030 | 0.552 ± 0.032, 27 | 0.069 ± 0.007 | gap | gap | gap | gap |
| F_s0, step 10 | 0.058, 4.44e-05 | 11 to 11 | -2.73 to -2.72 | 0.014 ± 0.002 | 0.386 ± 0.036, 27 | 0.150 ± 0.018 | gap | gap | gap | gap |
| F_s0, step 100 | 0.274, 2.09e-04 | 11 to 11 | -2.73 to -2.68 | 0.048 ± 0.004 | 0.495 ± 0.029, 23 | 0.038 ± 0.005 | gap | gap | gap | gap |

## Not run

- Steered per-position log P, and output entropy, at any state; so which tokens the vector raises, and whether either flattens the output, are unread.
- The steering share split by what the row holds of the question.
- A random vector matched to the steering vector on variance share.
- The template update's share at its saved steps 10, 20 and 50.
- Cosine between the vectors at different steps or seeds: the vectors are not saved.
- A held-out wording for the facts update; the rank read on the D3 maths rows; own-samples' second seed.
- Any D3 row on any of these files, and the 11-12 join.
