# What do three cheap measures say about the Qwen updates, on the same weights as the decomposition's rows? (E92)

We read three cheap measures on the saved Qwen updates at matched and full size, then read the steering vector again with output entropy saved, using the unchanged model at a range of temperatures as the reference. At the template update's size, the facts update and its vector both read as a flatter output; at full size, random vectors of the same norm and variance share earn as much as the vector does. For `GOALS.md` the steering share cannot be read across the two anchors as new against old at either size, and these reads describe the output's shape, not the cause in the weights.

*2026-11-05, `baseline-lead`. Plan `reports/plans/2026-11-05_plan_e92-comparators.md`, rules `experiments/qwen_comparators/results/predictions.json` and `predictions_amend1.json` (both unedited), notebook `notebooks/E92_qwen_comparators.ipynb`, figures `experiments/qwen_comparators/figures/`, from `plot.py`, after `dissect.py`.*

**Status: provisional, no independent check.** Every number is copied from `results/states/*.json`, `W0_*.json`, `controls.json`, or from `results/dissection.json` and `results/comparator_table.json`, which `dissect.py` writes from the saved per-position files on CPU. Those two files and everything they alone support were read after the run and were not filed. Amendment 1's numbers are copied from `results/amend1/`, under rules filed before its run; job 52467.

**Step 550 at seed 42 and step 500 at seed 142 are an unfiled addition.** They were not in the plan. Each is that seed's first saved state past E91's learning line. They are labelled wherever they appear.

**Names.** "Template update" is anchor 1, `anchor1_s0` and `anchor1_s1`. "Facts update" is anchor 2, the TOFU fine-tune; `anchor2_s0` is seed 42 and `anchor2_s142` is seed 142.

## The answer

No cheap number here tells the template update from the facts update. At the template's size, one steering vector recovers 0.83 of the facts update's gain. That is largely a flatter output: the unchanged model, made equally unsure, earns about three-quarters of the vector's rise, and no more right tokens come first. At full size, random vectors of the same size earn as much as the vector does. So the steering share cannot be read across anchors as new against old.

## What we were trying to test

The plan needs a cheap non-decomposition reading beside every D3 row, on the same state, seed and step. A row whose separation a comparator matches says less than its number suggests. Three measures are read, cheapest first, and none reads a decomposition.

- **Update size.** `norm` is the root of the summed squares over the 196 decomposed matrices, `ratio_norm` is that over the unchanged model's norm, 1310.517, and `share_squared` is its square. Do-nothing reads 0. Blind to direction.
- **Rank.** The unchanged model's rank of each next answer token in the whole vocabulary, 1 being the top, with its log P. Two summaries: the answer's first token, and the teacher-forced mean over every answer token, end token included. Blind on the facts rows to anything but trained wording.
- **One steering vector.** The vector is the mean over one half of the rows of the state's block input minus the unchanged model's, at answer positions. It is added to the unchanged model at every answer position on the other half. The share is the rise in log P over the state's gain. The block is the one with the largest first-half share.

**Controls.** The zero update must read as the unchanged model, and ran first and alone: largest log P difference 0.0, ranks equal, no block chosen. The template's update must raise its own tokens: gain 0.666 ± 0.051, defined and positive. The norm must equal E91's stored read within 1e-9: it does on every state E91 read. A random vector at the vector's norm, a Gaussian mix of the unchanged model's centred inputs, asks whether any vector of that size raises log P.

**Rows.** The template update: the prompted model's greedy answers on held-out problems, 64 x 256. The facts update: its trained rows in TOFU's own wording, 128 x 256. The GSM8K arms: gold solutions, 103 x 256, because the D3 maths rows hold the unchanged model's own answers. These are different rows per arm, so a difference between anchors can come from the rows as well as from the update.

**The amendment's stop rule.** A column counts only if the gap between the anchors' means exceeds the larger same-anchor seed spread.

**Amendment 1: a second read of the vector.** The first read could not say why one vector recovered 0.83, because entropy and steered positions were not saved. This read saves both, at the template's step 100 and the facts update's steps 29, 30 and 1250, on the scoring half of the rows, split by what the row holds of the question.

- **The reference for a flattening.** The unchanged model at the output temperature whose mean entropy equals the treatment's. A temperature keeps every rank, so this is what a pure flattening earns at that entropy. A temperature of 1 reads as the unchanged model.
- **The ceiling.** The temperature on the grid, 0.505 to 3.92, that earns the most mean log P at the unchanged model. It is the most a pure flattening could earn on those rows.
- **The word "flattens".** Entropy rises, and the rise in log P lies within a factor of two of the reference, each side clear by 2 standard errors.
- **The word "style".** The rise exceeds twice the reference, and the target token's identity predicts over half of the remainder out of sample. The template update, whose answers all open on "Given", is the rule's positive control.
- **The word "neither".** Neither pattern holds at those bars. It names no third cause.
- **Three new random vectors,** matched to the steering vector on both norm and variance share. `vector_beyond_matched` asks whether the vector's rise clears theirs by twice the combined errors and spread. False means a random direction of this norm and variance share earns the same, so the rise does not belong to the vector's direction.
- **The template's earlier steps:** 10, 20 and 50 at seed 0, and 10 at seed 1, through E92's own steering read.

What the rules read is the output's shape, not the cause in the weights. The filing says so, and says these reads do not touch the definition of new against reuse.

## What we expected, and why

The leans are in the scorecard. In words: the template's vector recovers a modest part and the random vector none. The facts update at full size gives under 0.05, because one vector cannot hold 787 facts, and ends above log P -0.2 on rows it trained. At matched size it gains under half a nat with a share undefined or under 0.3, because E91 reads no facts there. The GSM8K arms gain under 0.1 on gold tokens. At matched size only the first token's rank counts.

For Amendment 1, the leans were that the facts update and its vector read "flattens" at matched size, and that at full size the state sharpens and reads "neither". The template state was to read "style". The template's share was to fall with step, with low confidence because its steps are close in size.

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
- **At the full update the vector barely beats random in nats:** 0.25 against 0.18. The filed test still reads it above random on every state, but Amendment 1's random vectors, matched on variance share as well, earn -0.17 to 0.28 there, and the vector is not beyond them.
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

**Reading anchor 1 against anchor 2.** The template's 0.185 is at norm 0.487, which is its full update and the reference for "matched". Amendment 1 reads its earlier steps, below, and its share falls with step too. Equal norm is not equal effect on these rows. At equal norm the facts update's vector is about 3.5 times the template's and its gain 3 times larger. The unchanged model also starts lower on facts rows, -4.25 against -1.54. As a share of that shortfall the two gains are close, 0.43 and 0.48, and 0.89 at the full update; that ratio is post hoc.

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

**What the vector shows.** It is a constant added to block 27's input at 18% of that input's root mean square, against 5% for the template. It recovers 0.827 and 0.829 of the gain, and 0.807 and 0.822 on first tokens. Its variance share among the unchanged model's inputs is 0.076 against 0.013 for E92's random vector, which is unmatched; Amendment 1 adds vectors that match it. The random vector at that norm adds 0.22 nats. Added nats rise with norm across the six facts states' random vectors, 0.163 at 466 to 0.227 at 589; six points on one row set.

**What the first read inferred.** It said this was the shape of a flatter output distribution: less sure where it was right, much less wrong where it was wrong, and little change in order. Amendment 1 reads that directly, in the next section. That a random vector raises log P on these rows and lowers it on template rows fits a disturbance earning log P where the unchanged model is confidently wrong, which stays an inference.

**What the steering share is blind to.** One mean vector cannot express an input-dependent change, so a low share does not say which mechanism is new. A high share does not say the mechanism is reused: it is a share of mean log P, and this state raised mean log P without ranking more tokens first. It adds at every answer position, including after earlier steered ones. It is a comparator, not the definition of new against reuse: that definition, in `decisions.md`, names a capability and removes it, and this reads neither.

### Is the matched share reading a flatter output? By the filed rule yes at matched size, no at full size

![](../experiments/qwen_comparators/figures/temperature_ceiling.png)

**At matched size, state and vector both read "flattens" on the pool, on both seeds, so `share_reads_flattening` is true. At full size it is false.** The figure's black line is the ceiling, and each marker is a rise on the same scoring-half positions. All numbers below give seed 42 then seed 142, or seed 0 then seed 1.

**The controls were clean.** The steering and random shares repeat E92's with a difference of 0 on both template seeds. The unchanged model repeats E92's positions exactly, temperature 1 repeats the unchanged model, and the zero state reads "no rise" with no vector.

**How much temperature alone can earn.**

| row set | best temperature, not at the grid's edge | ceiling, nats | state's gain on these rows | ceiling over gain | unchanged model: mean surprise, entropy |
|---|---|---|---|---|---|
| Template, scoring half | 1.48 | 0.212 ± 0.034 | 0.586, 0.588 | 0.36 | 1.42, 0.48 |
| Facts, scoring half | 2.29 | 1.566 ± 0.062 | 2.095, 2.106 at matched size | 0.75, 0.74 | 4.30, 0.63 |

The ceiling is high on the facts rows because the unchanged model is confidently wrong there: its mean surprise is 4.30 nats against an entropy of 0.63. On template rows the ceiling is reachable, yet that update's entropy does not rise: the state's entropy change is -0.011, so its 0.59 nats do not come from flattening.

**Per site, on the pool and by what the row holds of the question.** Each cell gives both seeds. "Reference" is the unchanged model's rise at that treatment's own entropy.

| site | positions, share of site | state: word, rise | state: entropy change | state: reference | vector: word, rise | vector: entropy change | vector: reference | vector's share | beyond matched random |
|---|---|---|---|---|---|---|---|---|---|
| facts, matched, pool | 8,297 | flattens, 2.095, 2.106 | 1.233, 1.275 | 1.468, 1.477 | flattens, 1.732, 1.745 | 0.814, 0.841 | 1.290, 1.306 | 0.827, 0.829 | true |
| whole question | 6,903, 83% | flattens, 2.280, 2.292 | 1.251, 1.291 | 1.778, 1.783 | flattens, 1.884, 1.898 | 0.696, 0.720 | 1.589, 1.605 | 0.826, 0.828 | true |
| question cut | 722, 9% | flattens, 2.127, 2.122 | 1.742, 1.817 | 1.668, 1.680 | flattens, 1.794, 1.803 | 1.392, 1.432 | 1.576, 1.590 | 0.843, 0.850 | true |
| fragment | 672, 8% | flattens, 0.161, 0.176 | 0.504, 0.535 | 0.137, 0.144 | no rise, 0.112, 0.108 | 1.408, 1.449 | 0.228, 0.229 | 0.694, 0.612 | false |
| facts, full, pool | 8,297 | style, 3.809, 3.808 | -0.344, -0.335 | -2.722, -2.580 | neither, 0.246, 0.256 | 0.023, 0.031 | 0.081, 0.109 | 0.065, 0.067 | false |
| whole question | 6,903, 83% | style, 4.024, 4.029 | -0.276, -0.267 | -3.805, -3.805 | flattens, 0.284, 0.289 | 0.034, 0.037 | 0.222, 0.239 | 0.071, 0.072 | false |
| question cut | 722, 9% | neither, 3.814, 3.795 | -0.358, -0.347 | -2.325, -2.195 | flattens, 0.249, 0.287 | 0.057, 0.070 | 0.180, 0.218 | 0.065, 0.076 | false |
| fragment | 672, 8% | neither, 1.591, 1.549 | -1.020, -1.019 | -0.644, -0.643 | no call: neither, -0.153; no rise, -0.122 | -0.133, -0.080 | -0.055, -0.033 | -0.096, -0.079 | false |
| template, step 100, pool | 4,546 | style, 0.586, 0.588 | -0.011, -0.010 | -0.014, -0.012 | style, 0.109, 0.107 | 0.039, 0.038 | 0.039, 0.038 | 0.185, 0.182 | true |
| whole question | 1,254, 28% | style, 1.250, 1.252 | -0.030, -0.030 | -0.394, -0.397 | neither, 0.164, 0.162 | 0.010, 0.010 | 0.086, 0.085 | 0.131, 0.129 | true |
| question cut | 1,133, 25% | style, 0.881, 0.882 | 0.019, 0.020 | 0.100, 0.104 | flattens, 0.178, 0.176 | 0.025, 0.024 | 0.128, 0.127 | 0.202, 0.199 | true |
| fragment | 2,159, 47% | style, 0.046, 0.048 | -0.016, -0.013 | -0.006, -0.005 | neither, 0.040, 0.039 | 0.063, 0.062 | 0.018, 0.018 | 0.875, 0.810 | true |

- **Largely, not wholly.** At matched size the reference earns 1.47 of the state's 2.09 and 1.29 of the vector's 1.73, about 0.70 and 0.75. The rule's factor of two is wide, so a mix can read "flattens". The remainder is 0.63 nats on the state, and the token's identity predicts 0.51 of it out of sample. On trained rows that may be repeated wording, which the filing says the rule cannot tell apart from a style. We name no cause for it.
- **The state rises past the ceiling.** The state's 2.09 is 0.53 above anything a temperature earns, and the vector's 1.73 is 0.17 above it.
- **Ranks barely move.** The share of positions ranking the target first changes by +0.012 and +0.015 under the state and by -0.003 under the vector. A pure flattening leaves it at 0. At full size the state's change is +0.37.
- **Fragments differ.** On answers with no question in the row, the state rises only 0.16 and still reads "flattens". The vector there raises entropy most, by 1.41, with no clear rise and is not beyond random. So the vector and the state do not read alike on this class. The class is 8% of the facts positions and 47% of the template's, so the pools mix them in very different proportions. At full size the vector's pooled "neither" sits over "flattens" on both classes with a question.
- **The share is similar on the facts update's classes with a question and not on the template's.** Facts at matched size reads 0.83 and 0.84 on those two, and about 0.65 on fragments, where its rise is lost in noise. The template reads 0.13, 0.20 and 0.88, the last on a gain of only 0.046 nats. Its pooled 0.185 is a mix of these, so it is not the template's share on whole answers.

**What a vector alone does here, by site.**

| site | vector's rise | three matched random vectors' rises | their words | `vector_beyond_matched` |
|---|---|---|---|---|
| facts, matched, pool | 1.732, 1.745 | -0.14, -0.18, 0.34 | neither, neither, flattens | true |
| whole question | 1.884, 1.898 | -0.14, -0.16, 0.37 | neither, neither, flattens | true |
| question cut | 1.794, 1.803 | -0.08, -0.24, 0.33 | no rise, neither, flattens | true |
| fragment | 0.112, 0.108 | -0.17, -0.33, 0.03 | no rise, neither, no rise | false |
| facts, full, pool | 0.246, 0.256 | -0.13, -0.17, 0.28 | neither, neither, flattens | false |
| whole question | 0.284, 0.289 | -0.13, -0.15, 0.31 | neither, neither, flattens | false |
| question cut | 0.249, 0.287 | -0.07, -0.22, 0.29 | no rise, neither, flattens | false |
| fragment | -0.153, -0.122 | -0.12, -0.27, 0.04 | no rise, neither, no rise | false |
| template, step 100, pool | 0.109, 0.107 | 0.03, -0.02, 0.01 | flattens, neither, style | true |

Random rises are seed 42's and seed 0's; the other seed's are within 0.02.

- **The word "flattens" is not special to the vector.** One of three random vectors, `random_var_2`, reads it at both sizes. Two of three lower log P instead.
- **At matched size the amount is special to the vector.** It rises five times as far as the best random vector, 1.73 against 0.34, and raises entropy twelve times as far, 0.81 against 0.07. So on the pool, and on both classes with a question, the rise belongs to the vector's direction. On fragments it does not.
- **At full size no pool's rise belongs to the vector's direction.** The best random vector earns 0.28 and 0.27, more than the vector's 0.25 and 0.26. So the 0.065 share does not say what this vector carries.
- **Same size, very different effect.** The vector at step 29 and at step 1250 is nearly the same size, with norms 580 and 504 and variance shares 0.076 and 0.078. Its entropy change falls from 0.81 to 0.02. How the two directions relate is unread, as the vectors are not saved.
- **These are few directions.** The same three seeds are used at every site, so `random_var_2` at matched and full size may be nearly one direction, and the two seeds of a site are not independent draws. Cosines are not saved.
- **The norm-matched random vector reads "style" on the facts pool at both sizes.** It rises 0.22 and reads "flattens" on whole questions, "no rise" on cut ones and "neither" on fragments, so its "style" holds in no class alone. It has a variance share of 0.013. `style_rule_powered` is true, because the template state reads "style" on both seeds, but the word is not specific: the template's own vector and one of its random vectors read it too. So the facts state's "style" at full size is not read here as a shift to a style, and on trained rows it may be repetition.

### Does the template's own share fall with step? Yes, though that does not join the two updates into one curve

| template update | step 10 | step 20 | step 50 | step 100 |
|---|---|---|---|---|
| norm | 0.128 | 0.303 | 0.463 | 0.487 |
| seed 0: share ± se | 0.424 ± 0.052 | 0.359 ± 0.043 | 0.209 ± 0.022 | 0.185 ± 0.019 |
| seed 0: gain on scoring half, vector's nats | 0.440, 0.187 | 0.570, 0.204 | 0.589, 0.123 | 0.586, 0.109 |
| seed 0: random share | -0.055 | -0.050 | -0.039 | -0.038 |
| seed 1: share ± se | 0.419 ± 0.051 | not saved | not saved | 0.182 ± 0.019 |

Block 27 is chosen at every step. The filed rule compares each seed's step 10 with its step 100, and both read "falls"; steps 20 and 50 are reported, not tested. The numerator falls as well from step 20, 0.20 to 0.11 nats, while the gain holds near 0.58.

**This does not make them one curve.** The template's steps span norm 0.128 to 0.487. The facts line starts at 0.477 and runs to 18.1. So "falls" on both describes two lines that share almost no range of size, and they sit on different rows. In their one shared range, near norm 0.48, the template reads 0.19 and the facts update 0.83. The filing adds that the template's KL to start is nearly level from step 20, so its steps 20 to 100 are close in that sense too.

### What does this do to the steering-share column?

At matched size, a share of 0.83 is largely a flattening of the output and is not a reading of what is being learned. The steering-share column is not to be read across anchors at this size as new against old. The filed rule sets it beside a D3 row as no evidence on mechanism. The same read applies to the state's gain at this size, as a description; the rule's note goes on the share cells only.

**Other sizes, by the filed rules.**

- **Full size, facts update.** The flattening note does not apply, since `share_reads_flattening` is false. But `vector_beyond_matched` is false on all four pools, so the 0.065 does not belong to the vector's direction. It can be read only as this: one vector does no better here than a random vector of its size. That is no statement about new against old, by the blind line above.
- **Template, step 100.** Its vector's rise clears the matched random vectors on all four pools, and its state does not flatten. So 0.185 stands as a reading of that vector at that step, labelled by step. It falls with step and is a mix of very different classes.
- **Across anchors.** Every comparison across anchors has on its facts side either a matched cell, which reads a flattening, or a full-size cell, which does not beat a random vector. So there is no size in these reads at which the column can be read across anchors as new against old. The filing never offered one: even a clean read describes the output's shape.
- **Not read by these rules.** The facts update's steps 550 and 500 and all the GSM8K arms got no verdict. Their shares carry no note and no licence: several GSM8K shares are high, up to 0.91, and whether they read a flattening is unknown.

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

The steering share, the gain and the unchanged model's column all pool these, in different proportions per arm. On the scoring half that Amendment 1 reads, the template's classes are 28%, 25% and 47%, and the facts rows' 83%, 9% and 8%; its table above gives the share by class. This is the filed rule, and it is not a bug in E92's code; E92's rows match E57's and E86's by the plan's check.

### Do the GSM8K arms move gold tokens? Yes, past the filed band

![](../experiments/qwen_comparators/figures/gsm8k_gain.png)

At step 100 RL reads -0.277 and +0.163, the null 0.283 and 0.595, own-samples 0.048 on its one seed. The null gains most, so gold-token gain does not track a meaningful reward. The share ranked first was 0.686 unchanged and reads 0.683, 0.686, 0.675, 0.696 and 0.687. Answers with some of their question in the row carry the change. On whole-question positions the null's second seed reads -3.04 to -2.17 and RL's first seed -3.04 to -3.52, and fragments stay within 0.02. These read as description only. A gold-token gain is not the arm's accuracy, and this is not a claim about it.

## What we learned

For new against existing, these columns are comparators to set beside the decomposition's rows, and none of them decides it. A one-number steering share needs its step and size beside it, because it changes more than twelve-fold within one update on the same rows, reverses against the template, and falls with step on the template too. By the filed rule, the facts update's matched share and gain are largely a flatter output, so neither is evidence of learning, and its full-size share is no better than a random vector's. So the column cannot be read across anchors as new against old at either size. These reads concern the output's shape, and the cause in the weights is untouched. E91's call that the matched step is a size control only stands, and this supports it from a second direction. The decomposition's rows are still to be read.

## How compelling is the evidence

- **Seeds.** Two on each anchor, and they agree closely. Own-samples has one.
- **Rows.** Each arm is on its own rows, so anchor gaps mix row and update. Trained rows for the facts update and held-out problems for the template.
- **E92's random control** is matched on norm only. Its variance share runs from 0.013 to 0.401 and the vector's from 0.003 to 0.085. It beats by the filed test on all 18 states. Amendment 1 adds three vectors matched on variance share too, at six sites only.
- **The readings of the matched state are per position and agree across seeds.** "Flattens" is now read by a filed rule with a positive control, the ceiling, which reads "flattens" on every class of both row sets. The rule's factor of two is wide, and one temperature serves a whole pool.
- **"Style" is weak.** Its positive control holds, but a random vector also reads it on the pool, so the word does not exclude a disturbance.
- **Three random directions are a small null,** reused across sites.
- **What the rules are blind to.** They read the output's shape, not the cause in the weights, and only target tokens' log P with whole-vocabulary entropy.

## Surprises

**Something unexpected happened that I cannot explain.** We predicted the facts update's share at matched size would be undefined or under 0.3, because that state had learned nothing, and it read 0.83 on both seeds. We predicted its gain under half a nat, and it read 2.05 and 2.06. The section above says what the files show. Amendment 1 finds that this is largely a flatter output, by the filed rule. Why one constant vector produces that flattening, and what the remaining 0.44 nats of its rise are, is not explained.

- **Log P at the full update, -0.48 against a filed above -0.2.** The miss stands. The files give its cause: positions with the whole question read -0.12, those with it cut about -1.2, fragments about -3.0. Fragments' end tokens read -14.9 and -15.4, against -0.02 and -0.01 on all other answers.
- **The first token's rank worsens while its log P rises,** at matched size on both seeds.
- **The vector is largest at the smallest update:** 580 and 589 at steps 29 and 30, against 504 and 490 at step 1250.
- **The vector's variance share is not monotone in step:** 0.076, 0.007, 0.078 at seed 42.
- **The template's vector recovers less than random on first tokens:** 0.021 against 0.048. Its share is carried by later tokens; the first token's 15-nat gain is not reached by the vector.
- **All gain-side leans for GSM8K missed,** with row standard errors far smaller than the lean assumed.
- **The unchanged model is more confident than filed on the facts rows.** Its entropy reads 0.63 against a lean of 1 to 4 nats, with a mean surprise of 4.30, which is why a temperature earns so much there.
- **The state reads "style" at full size, against a lean of "neither".** The word is weak here, since a random vector reads it too.
- **A random vector matched on variance share earns more than the steering vector at full size:** 0.28 against 0.25 at seed 42. Two of the three lower log P on the facts pool at every site, against a lean that they would rise.
- **The vector raises entropy most on fragments at matched size,** 1.41 nats, where its rise is lost in noise.
- **The template's vector reads "style" on the pool, against a lean of "neither",** and reads "style" in no class alone.

## What next

1. **Fill the comparator table with its notes on.** `qwen-decomp-lead`'s D3 rows go beside these cells, and the matched share cells go as no evidence on mechanism.
2. **The same read on the GSM8K arms and steps 550 and 500, only if one of their shares is to be set beside a D3 row as evidence.** It would be one `debug` job at $0, and E92's last amendment. If state and vector read "flattens" there, the note extends to those cells.
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

**Amendment 1, filed leans and rules, unedited.** Read on both seeds unless noted.

| claim | rule | read | verdict |
|---|---|---|---|
| Unchanged model repeats E92, temperature 1 repeats it, shares repeat | stop on a miss; shares within 0.00312 | 0 off on each | held |
| Zero state | "no rise", no vector | "no rise", no vector | held |
| Template state's rise | about 0.586, 0.588 | 0.586, 0.588 | held |
| Template state's word | lean medium: "style", entropy change near or below 0 | style; -0.011, -0.010 | held; rule powered |
| Template vector's word | lean low: "neither" | style | missed |
| Template rows' entropy | lean low: 0.3 to 1.5 | 0.48 | held |
| Facts rows' entropy | lean low: 1 to 4 | 0.63 | missed |
| Template ceiling | lean medium: under 0.3 | 0.212 | held |
| Facts ceiling | lean low: at least 1 | 1.566 | held |
| Matched, rises | about 2.09, 2.11; vector 1.73, 1.75 | 2.095, 2.106; 1.732, 1.745 | held |
| Matched, state's word and entropy | lean medium: "flattens"; above 0 | flattens; 1.233, 1.275 | held |
| Matched, vector's word | lean medium: "flattens" | flattens | held |
| Matched, random vectors' rise | lean low: above 0, far under the vector's | -0.14, -0.18, 0.34 | missed on above 0; held on far under |
| Matched, `vector_beyond_matched` | lean high: true on the pool | true | held |
| Matched, fragments | lean medium: rise near 0 under both | state 0.161, 0.176, clear of noise; vector 0.112, 0.108, in noise | held on size |
| Full, rises | about 3.81; vector 0.25 | 3.81, 3.81; 0.246, 0.256 | held |
| Full, state's entropy | lean high: below 0 | -0.344, -0.335 | held |
| Full, state's word | lean medium: "neither" | style | missed |
| Full, vector's word | lean low: "flattens" or "no rise" | neither | missed |
| Full, `vector_beyond_matched` | lean low: false on the pool | false | held |
| Template's share with step | lean low: "falls", on both seeds | falls | held |
| Template's gain at step 10, block | lean medium: positive; lean low: block 27 | 0.521, 0.519; 27 | held |
| `share_reads_flattening` | lean medium: true at matched, false at full | true, false | held |

**Reading the files.** In `readings.json`, `flatten_ceiling`'s `share_of_state_gain` reads 1 because the code passes the ceiling as its own state; it is not a result. The 0.75 above is the ceiling over the state's gain, computed here.

**Numbers from the runner, checked.** All match the files. One rounding: the template's random reads -0.0378 at seed 0 and -0.0375 at seed 1, given as -.037. The runner's list named two misses; the files give eight, with T5, A4, M1, M2, G3 and S2 added.

## Departures from the literature

- **Steering.** The vector's definition is carried from this project's `steering.py`, E52 and E54, which adds at one answer position. Here it adds at every answer position, because a first token alone can carry little of the gain on the facts rows. The vector is a difference between two models on one input, not a contrast between two behaviours on one model.
- **Random control.** A Gaussian mix of the model's own inputs, not an isotropic direction, which would read 0 by construction. Matched on norm, not on variance.
- **TOFU, Maini et al. 2024.** Read on its trained rows, packed in rows of 256, by mean log P, and not by TOFU's Truth Ratio or forget metrics. The 13 rows E84 dropped stay as context and mark no answer.
- **GSM8K arms.** Gold solutions, not the D3 maths rows, which hold the unchanged model's own answers.
- **New against reuse, Jain et al. 2024 by way of `decisions.md`.** Not tested here: no capability is named and nothing is removed.
- **Temperature, Amendment 1.** Temperature scaling, as in Guo et al. 2017, fits one temperature on held-out data to calibrate a model. Here it is a reference and fits nothing: the temperature is set to match a treatment's entropy on the same positions, and the ceiling is chosen on the rows it is read on. The filing credits no source.

## Provenance

Jobs 52271, 52273, 52284 and 52382, all in `results/run.log`, cluster L40, `debug` QoS, $0. Per-position reads in `results/large/`, 118 KB to 215 KB a state. `dissect.py` and `plot.py` run on CPU after the run and load no model. `anchor2_s0`'s steps 29 and 550 and `anchor2_s142`'s are E91's retrained paths; E91's rerun against saved step 100 reads 0.0074 of its norm at seed 42 and 0.0 at seed 142, against a seed distance of 0.90, so they lie on the full update's path. Amendment 1: job 52467, `results/amend1/run.log`, one L40, `debug` QoS, $0, 371 s, peak 23.0 GB on the card and 11.0 GB of host; per-position files in `results/large/amend1/`.

## The comparator table for the D3 rows

Same state, seed and step where a row exists. **Every D3 cell is a gap:** no D3 row has been read on any of these files. The rows are `qwen-decomp-lead`'s, due 11-07 to 11-09 on the anchors and 11-10 to 11-11 on the GSM8K arms. Own-samples has no second seed, so it has no row to fill. E75's clean table is a D1 read, not a D3 row, and is not joined here. Machine copy: `experiments/qwen_comparators/results/comparator_table.json`, written before Amendment 1 and left as it is, so the notes below are here only.

**Notes the filed rules allow on the steering-share column.** No rule adds a column.

- **a. Matched size.** `share_reads_flattening` is true: this share reads a flatter output, and it goes beside a D3 row as no evidence on mechanism. It is not to be read across anchors as new against old.
- **b. Template, labelled by step.** Share at seed 0: 0.424 at step 10, 0.359 at 20, 0.209 at 50, 0.185 at 100. Seed 1: 0.419 at step 10, 0.182 at 100. This cell is step 100 only.

| state | size: norm, ratio to W0 | first token, median rank, W0 to state | mean log P, all tokens, W0 to state | gain ± se | steering share ± se, block | random share ± se | route split | δW against null | kept under masks | think overlap |
|---|---|---|---|---|---|---|---|---|---|---|
| anchor1_s0, step 100 | 0.487, 3.71e-04 | 21 to 1 | -1.54 to -0.87 | 0.666 ± 0.051 | 0.185 ± 0.019, 27, *note b* | -0.038 ± 0.007 | gap | gap | gap | gap |
| anchor1_s1, step 100 | 0.489, 3.73e-04 | 21 to 1 | -1.54 to -0.87 | 0.666 ± 0.051 | 0.182 ± 0.019, 27, *note b* | -0.037 ± 0.006 | gap | gap | gap | gap |
| anchor2_s0, step 29 | 0.477, 3.64e-04 | 1 to 2 | -4.25 to -2.20 | 2.052 ± 0.045 | 0.827 ± 0.025, 27, *note a* | 0.106 ± 0.007 | gap | gap | gap | gap |
| anchor2_s142, step 30 | 0.496, 3.78e-04 | 1 to 2 | -4.25 to -2.19 | 2.062 ± 0.046 | 0.829 ± 0.025, 27, *note a* | 0.108 ± 0.008 | gap | gap | gap | gap |
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

- Amendment 1's read on the GSM8K arms and on the facts update's steps 550 and 500, so their share cells have no verdict.
- Any read of the cause in the weights: the rules read the output's shape.
- Which direction inside the vector does the work; the vector at any block but the one E92 chose.
- More than three matched random directions per site; new seeds per site.
- A per-position entropy match: the temperature reference matches entropy in the mean over a pool.
- The template's steps 2, 3 and 5, saved but outside E91's grid; seed 1's steps 20 and 50, not saved.
- Rows that share a size range and a row set across the two updates: the template and facts lines stay on different rows.
- Cosine between the vectors at different steps or seeds: the vectors are not saved.
- A held-out wording for the facts update; the rank read on the D3 maths rows; own-samples' second seed.
- Any D3 row on any of these files, and the 11-12 join.
