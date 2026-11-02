# Analysis: do fine-tunes built to elicit sit on fewer MLP neurons than those built to teach something new? (E63)

*2026-11-02, coding agent, baseline-lead E63 writer. Plan: `reports/plans/2026-11-02_plan_e63-neuron-sparsity.md`, amendments 1-19. Rules: `experiments/neuron_sparsity/results/predictions.json`. Code review: `reports/reviews/2026-11-02_review_e63-code.md`. Explanation: `notebooks/E63_neuron_sparsity.ipynb`. Provenance at the end.*

We read five neuron measures on saved fine-tunes of three families, comparing those built to elicit with those built to teach something new. None of the five tells the two kinds apart on all the pairs, because the gap's direction flips between families and the main pair's gap in carrying neurons closes with progress. So counting neurons does not give what the decomposition is claimed to give, at this size and on these pairs, and neither reading yet has a causal check.

## The answer

No. No reading of neurons tells the two kinds apart on all the pairs we tried. A finished update is spread almost evenly over the neurons, whichever kind built it, and no fine-tune rescales the neurons the original model already has. Activation shifts and the neurons that carry the gain sit narrower for elicit on two of the three families, and wider on the third. On the main pair, the gap in carrying neurons closes once the cipher has learned as much.

The filed tree says: none answers yes, so write up, report an elicit and new gap under the matched-size test alone as a progress effect, and leave the truth-set reading to the user's say. I recomputed every test, reference gap and guard from the per-state values in `weights.json`, `gradients.json` and `windows.json`; all agree with `score.json`, so "none answers yes" stands. One wording point: no number has a gap under the matched-size test *alone*. The activation spread passes it and also family C, and the count fails it on the floor. The section on the count says which gap reads as progress.

## The question, and what "built to elicit" means

The project asks whether a VPD decomposition can tell a fine-tune that strengthens something the original model does from one that teaches something new. If simply counting MLP neurons did the same, the decomposition would have a cheap rival.

"Eliciting" here means **built to elicit**, not the causal label of `decisions.md` row 112. Row 112 defines elicitation mechanistically: the fine-tuned behaviour runs through a mechanism the original model already has. These pairs have no causal check. Every "elicit" below is a statement about how a task was built.

## What was measured

Nothing was trained. A neuron is one of the 12288 GELU outputs across the four blocks. A neuron's update is its row of the change in `c_fc`, its input weights, plus its column of the change in `down_proj`, its output weights. All values are read on the target model in fp32.

**States.** 26 saved full-weight fine-tunes, plus one update that learned nothing.

| family | built to elicit | built to be new | notes |
|---|---|---|---|
| A, main pair | elicit map, seeds 0-4, step 10 | cipher, seeds 0-4, step 9, same update size | same keys and windows; the cipher at its own learned step 30, seeds 0-1, is the equal-progress state |
| B | same word back, seeds 0-1, step 10 | random word, seeds 0-1, step 10 | same-word and random tables on other keys, at steps 10 and 30, reported beside, unscored |
| C | lowercase, seeds 0-1, step 300 | 500-word cipher, seeds 0-1, step 300 | 50-word cipher, seeds 0-1, reported beside, unscored |
| reference | | | an update that learned nothing, on a code task with no answer position; weight readings only |

**The five readings.** Each has a filed reference and a positive control.

| reading, code | definition | sparser is | control and its reading | blind to | bears on |
|---|---|---|---|---|---|
| spread of the update, W1 | participation ratio over neurons of each neuron's squared update, $(\sum e_i)^2 / \sum e_i^2$, as a fraction of 12288 | lower | 16-neuron update: 0.0013, filed 16/12288. Unit Gaussian 0.9987, filed 0.9987. Rotated 16-neuron update 0.218, filed range 0.10 to 0.25 | attention and read-out matrices; Adam's flattening | whether the update itself is on fewer neurons |
| spread of the first gradient, W1g | the same on the first-batch gradient of the task's loss at the original weights | lower | same function as above. Reference: the same batch with targets shuffled | later steps, what Adam then does, progress; on family C it may mostly read ordinary text | whether the task's own direction is on fewer neurons, before Adam |
| rescale share, W2 | share of the update's energy along each neuron's own original row and column, energy-weighted | higher | rescaling 16 neurons: 1.000. Gaussian rows 0.00130, columns 0.00129, against 1/768 = 0.00130 | a neuron rotating towards another neuron | whether the fine-tune re-weights existing neurons |
| spread of the shift, A1 | per window, participation ratio of the squared tuned minus original activation at the answer position; median over the scoring windows, as a fraction | lower | 16-neuron update: 0.00056, filed at most 0.0013 | neurons that move and do nothing | whether activity changes on fewer neurons |
| neurons carrying the gain, C1 | per window, rank neurons by that window's own shift; top k get tuned activations at every position of an original-model forward; log2 k where the share first reaches half of the state's own full share, interpolated | lower | 16-neuron update: 1.99, filed at most 4; rise at k = 16 equals rise at k = 12288 exactly | changes to a neuron's output weights, which stay at the original values | whether the gain causally needs few neurons per window |

The **gain** is the mean rise in log probability of the target, tuned over original. A **share** reads 0 for the original model and 1 for the fine-tuned model.

**The tests.** Separation is E51's rule: the gap between the kinds' means is strictly over the larger within-kind range, and elicit is on the sparser side. On the count the spread is at least one grid step, 1 in log2 (Amendment 13). In every test, the sparser-signed gap must also beat the same gap on each reference (Amendment 14).

- **Matched size,** T1: family A, five seeds.
- **Equal progress,** T2: elicit seeds 0-1 against the cipher at its own step.
- **Family B,** T3_B, and **family C,** T3_C, each separately.
- **Guard:** on the five elicit states, the mean lies on the sparser side of the reference's mean by more than the number's range. References: rotated value for the update's spread; shuffled-target gradient for the gradient's; per-state Gaussian for the rescale share; both a Gaussian update and the rotated value for the shift; a random order for the count.

A number answers yes only if all four tests and the guard pass.

**How to read a pass (Amendment 15).** T1, T2 and T3_C compare one task with one task. Between seeds a number barely moves, so two different tasks almost always separate and only the direction carries information. A pass reads "these tasks differ, in this direction". A simulated false yes with task differences ten times seed noise and no kind effect is about 2% per number, about 9% over five.

**Disclosure.** While reviewing the code, the reviewer computed the update's spread and its rotated value on all 26 states before the amendments were filed. So that number's outcome, no yes and elicit denser in every family, was known before they were. The rule changes were chosen on what the rules can do and do not change that outcome. No other number was read.

## What we expected, and why

All leans are medium at most. Each is scored at the end as filed.

- **The update's spread will not answer yes.** Adam's per-element normalisation flattens any update. Seen before the amendments, so not blind.
- **The first gradient's spread: no lean.**
- **The rescale share will not answer yes.** No reason to expect a few Adam steps to align with each neuron's own weights.
- **The shift's spread: no lean.**
- **The count passes at matched size and fails at equal progress,** as E56's coarse count did.

## Controls, run first and alone

Stage 0 passed, first and alone. All 33 files matched their sha256. The identity patch read exactly 0 on all nine tasks, tolerance 1e-6. The mean-ranked path reproduced E56's sixteen-neuron share exactly on `elicit_s0` and `cipher_s0`. The per-window path's full-width share was within 1e-6 of E56's, tolerance 1e-3. All 24 first-batch losses matched the fine-tunes' logged step 1 within 0.01 nats, the stop rule; KL on the other positions read 0 on the check state. No SURPRISE line appears in the log.

The 16-neuron update has a negative gain, -1.53 nats, and a full-width share of 0.163. It is readable under the filed rule, which allows a negative gain with a positive share.

Reported expectations:

| filed expectation | read | |
|---|---|---|
| gains near 6.3, 10.5, 13.7 on elicit, cipher, own-step cipher | 6.32, 10.53, 13.73 | as expected |
| update that learned nothing: spread within 5% of its rotated value | 0.980 against 0.983 | as expected |
| the same update: rescale share within a factor of 2 of its Gaussian | 0.00164 against 0.00131 | as expected |
| random order's count near 12.5 | 11.82 on elicit, 12.37 on the matched cipher; all states 11.64 to 12.42 | a little lower, measured and not gated |
| per-block spread near 0.999 unless flattening is broken | about 0.98 on family A at matched size; down to 0.866 on the broad cipher's last block | lower than expected, reported, not gated |
| run time 8 to 12 minutes | 2 min 12 s | shorter |

## What we observed

### Is the update itself on fewer neurons when built to elicit?

![The spread of the finished update and of the first gradient per task, against references](../../experiments/neuron_sparsity/figures/update_sparsity.png)

**Reading it.** No: elicit is the wider side in every family, for the finished update and for the first gradient. Finished updates read near 1, the elicit map 0.976 and the matched cipher 0.970. An update that learned nothing reads 0.980, so being this flat says nothing about the task. Rotating the neuron index narrows the elicit map to 0.923, so on this task the neuron basis is wider than a random basis. First gradients are about twenty times narrower, 0.053 on elicit and 0.038 on the cipher, a gap under the seed spread. Shuffled targets read much the same, 0.043 on elicit, so this narrowness does not depend on the task's targets. Adam's flattening remains the filed explanation of the difference between the two readings; no SGD fine-tune was run to test it. Why the first gradient is narrow was not tested either: the review's suspicion is the GELU's factor in a gradient row.

Spread of the update, per state, means within a task:

| task | spread | rotated | Gaussian, per state | range over seeds |
|---|---|---|---|---|
| elicit map | 0.9761 | 0.9231 | 0.9914 | 0.0051 |
| cipher, same size | 0.9704 | 0.8632 | 0.9881 | 0.0035 |
| cipher, own step | 0.9225 | 0.9089 | 0.9690 | 0.0106 |
| same word back | 0.9744 | 0.8853 | 0.9951 | 0.0003 |
| random word, step 10 | 0.9699 | 0.8802 | 0.9878 | 0.0031 |
| same word, other keys, beside | 0.9725 | 0.8869 | 0.9936 | 0.0026 |
| random word, other keys, step 30, beside | 0.9263 | 0.9149 | 0.9671 | 0.0031 |
| lowercase | 0.9379 | 0.9732 | 0.9782 | 0.0011 |
| 500-word cipher | 0.8162 | 0.8929 | 0.9069 | 0.0033 |
| 50-word cipher, beside | 0.8958 | 0.9527 | 0.9611 | 0.0061 |
| update that learned nothing | 0.9800 | 0.9830 | 0.9944 | one state |

The block readings: on family C the broad cipher puts 38% of its update's energy in the last block and lowercase 30%, and the last block's own spread reads 0.87 and 0.94. So part of family C's gap is how the update divides among blocks, as the review found. The rotated value keeps two thirds of that gap, 0.080 of 0.122.

Spread of the first gradient, means within a task:

| task | spread | shuffled targets | rotated | Gaussian, per state |
|---|---|---|---|---|
| elicit map | 0.0530 | 0.0433 | 0.520 | 0.674 |
| cipher, same size | 0.0382 | 0.0379 | 0.581 | 0.784 |
| same word back | 0.0395 | 0.0316 | 0.527 | 0.715 |
| random word | 0.0372 | 0.0281 | 0.560 | 0.752 |
| lowercase | 0.0152 | 0.0106 | 0.514 | 0.679 |
| 500-word cipher | 0.0124 | 0.0109 | 0.389 | 0.649 |
| 50-word cipher, beside | 0.0254 | 0.0127 | 0.563 | 0.733 |

The elicit map's seed range, 0.040, is large because seed 1 reads 0.0795. The equal-progress test on this number reads the same gradients as matched size at seeds 0-1, since a gradient at the original weights has made no progress (`tests.T2_W1g_note`); it adds nothing. The review's suggested follow-up for family C, reading the task's text minus the cased rows, was not run: the number already fails matched size and the guard, so it could not answer yes.

### Do fine-tunes rescale the neurons the original model already has?

![The share of the update along each neuron's own weights per task, against the Gaussian's and the rescaling control's](../../experiments/neuron_sparsity/figures/rescale_share.png)

**Reading it.** No task does. All the fine-tunes read 0.0014 to 0.0020, next to the Gaussian's 0.0013, while the control that rescales 16 neurons reads 1. Elicit beats its reference at equal progress and on family C, but those gaps are 0.0003 and 0.0001. On family A at matched size the cipher reads higher, 0.0020 against 0.0017. The more trained update reads lower on both tables that were read at two steps: the cipher 0.0020 and 0.0014, the random word 0.0020 and 0.0015. The equal-progress pass therefore compares a step-10 update with a step-30 one. The reading moves with training at least as much as with kind; the second pair is on other keys, so this is not one trajectory.

| task | rescale share | rows | columns | Gaussian |
|---|---|---|---|---|
| elicit map | 0.00175 | 0.00194 | 0.00154 | 0.00131 |
| cipher, same size | 0.00196 | 0.00247 | 0.00147 | 0.00131 |
| cipher, own step | 0.00143 | 0.00155 | 0.00128 | 0.00131 |
| same word back | 0.00199 | 0.00247 | 0.00150 | 0.00130 |
| random word, step 10 | 0.00199 | 0.00245 | 0.00156 | 0.00130 |
| same word, other keys, beside | 0.00200 | 0.00243 | 0.00155 | 0.00131 |
| random word, other keys, step 30, beside | 0.00145 | 0.00160 | 0.00129 | 0.00131 |
| lowercase | 0.00202 | 0.00239 | 0.00155 | 0.00130 |
| 500-word cipher | 0.00191 | 0.00230 | 0.00146 | 0.00131 |
| 50-word cipher, beside | 0.00155 | 0.00180 | 0.00125 | 0.00129 |
| update that learned nothing | 0.00164 | 0.00182 | 0.00140 | 0.00131 |

### Does the activation shift sit on fewer neurons when built to elicit?

![The spread of one window's activation shift per task, against a Gaussian update's, the rotated value and the control](../../experiments/neuron_sparsity/figures/activation_sparsity.png)

**Reading it.** On families A and C yes, on family B the reverse. The elicit map shifts about 300 effective neurons per window and the matched cipher about 370, a gap just over the seed spread. Lowercase shifts about 150 and the broad cipher about 340. On family B the same-word table shifts about 460 and the random table about 360, so elicit is the wider side there. This is the only reading that passes at matched size. It is not a progress effect, since the gap widens at equal progress: the trained cipher shifts about 530 neurons. That test fails only because the rotated value widens more, 0.048 against the number's 0.018. A Gaussian update of the same norms already shifts only about 650 neurons, so a low level alone says little about the fine-tune. The review's toy model puts that on the GELU; on the real model that cause was not tested. The trained cipher, at 0.043, sits near its Gaussian at 0.053.

| task | spread | Gaussian update | rotated | effective neurons |
|---|---|---|---|---|
| elicit map | 0.0246 | 0.0525 | 0.163 | 303 |
| cipher, same size | 0.0298 | 0.0522 | 0.155 | 366 |
| cipher, own step | 0.0430 | 0.0530 | 0.208 | 528 |
| same word back | 0.0375 | 0.0458 | 0.169 | 460 |
| random word, step 10 | 0.0294 | 0.0505 | 0.160 | 361 |
| same word, other keys, beside | 0.0406 | 0.0521 | 0.159 | 498 |
| random word, other keys, step 30, beside | 0.0445 | 0.0509 | 0.219 | 547 |
| lowercase | 0.0123 | 0.0459 | 0.159 | 151 |
| 500-word cipher | 0.0276 | 0.0482 | 0.143 | 339 |
| 50-word cipher, beside | 0.0353 | 0.0512 | 0.166 | 434 |
| control, 16 neurons | 0.00056 | 0.0243 | 0.084 | 7 |

The two ciphers of family C, both new, differ by 0.0077, half the kind gap of 0.0152.

### How many neurons carry the gain on the main pair?

![Share of the gain against the number of neurons patched, per-window ranks beside E56's mean ranks and a random order, on family A](../../experiments/neuron_sparsity/figures/c1_family_a.png)

**Reading it.** Fewer neurons than E56 found, for both kinds, and the elicit gap misses the filed floor by a hair, then closes with progress. Half of each state's own full share needs about 240 neurons on the elicit map and about 470 on the matched cipher. That gap is 0.965 in log2, under the floor of 1. It is a narrow threshold miss: the elicit seeds run 7.73 to 8.11 and the cipher seeds 8.78 to 9.09, so no seed overlaps. At equal progress it shrinks to 0.17, since the cipher at its own step needs about 260. The control reaches its half at about 4, and a random order needs about 3600 on elicit. About half of the matched-size gap, 0.55, is also there on the random order. Why that is was not tested; the two kinds' different full shares are one candidate.

**Comparison with E56.** E56 ranked neurons by their mean shift over all windows, and asked for half of the whole gain on a coarse grid. Its gap was 4096 against 12288, 1.58 in log2, and the own-step cipher read 4096. Three things differ here: ranks per window, half of each state's own full share, and interpolation. The curves show what the first and second do.

| k | elicit, per window | elicit, E56 | cipher same size, per window | cipher same size, E56 | cipher own step, per window | cipher own step, E56 |
|---|---|---|---|---|---|---|
| 16 | 0.110 | 0.004 | 0.026 | -0.013 | 0.059 | -0.014 |
| 64 | 0.248 | 0.032 | 0.085 | -0.014 | 0.207 | 0.010 |
| 256 | 0.479 | 0.148 | 0.246 | 0.048 | 0.484 | 0.095 |
| 1024 | 0.740 | 0.426 | 0.461 | 0.208 | 0.797 | 0.360 |
| 4096 | 0.887 | 0.750 | 0.607 | 0.455 | 0.944 | 0.765 |
| 12288 | 0.936 | 0.936 | 0.694 | 0.694 | 0.973 | 0.973 |

- **Ranking per window lowers the count on both kinds.** At 256 neurons the elicit map carries 0.48 of its gain, against 0.15 ranked by mean shift. E56's "the top sixteen carry about nothing" no longer holds per window on elicit, 0.11, while the cipher reads 0.03. That fits different keys using different neurons, the thing the plan said E56 could not see. The distinct neurons over all windows were not counted.
- **The target matters as much as the ranking.** The cipher's full share is 0.69 and elicit's 0.94, so half of its own share is a lower bar for the cipher. On E56's target of half the whole gain, read off these curves and **not filed**, elicit reads 8.15 in log2, about 280 neurons, and the matched cipher 10.51, about 1450. That gap is 2.36. The same reading gives 8.07 on elicit seeds 0-1 and 8.10 on the own-step cipher, a gap of 0.03.
- **The pattern matches E56.** At matched size elicit needs fewer, at equal progress they agree. Here it holds on the filed target and on E56's.

**Progress.** Under the tree this is the gap reported as a progress effect, with its limits: the count fell by 0.87 from the cipher's step 9 to its step 30. The random word on other keys at step 30 reads 0.81 below the random word at step 10, which is a different table, so not one trajectory. No elicit state at another step was read. E56's independent check also applies: at matched size, progress is itself a consequence of kind, so "progress and not kind" is not established.

### Does the same count hold on the other pairs?

![Share of the gain against neurons patched on families B and C, two panels with shared axes](../../experiments/neuron_sparsity/figures/c1_families_bc.png)

**Reading it.** No, the direction flips. On family B elicit is the wider side, about 580 neurons against 360, a gap of 0.70 under the floor. The second same-word table, on other keys, reads about 210. So two tables built the same way differ by 1.44 in log2, more than the kinds do on family B or on family A. On family C lowercase needs about 15 neurons and the broad cipher about 110, a gap of 2.90 and the only clear pass of this reading. The 50-word cipher reads about 120, 0.17 from the broad cipher, so the two ciphers agree. Still, this test compares one elicit task with new tasks of one kind of build, and a pass reads "these tasks differ, in this direction". Lowercase also gains about half of what the broad cipher does, 1.57 nats against 3.27, and whether that matters was not tested. Why family B reverses is not established. Two facts sit beside it, neither tested as a cause. The same-word table starts at a target probability of about 2.5e-4, where the elicit map starts near 0.03. All neurons together carry only 0.56 of its gain, against the random word's 0.77.

| task | count, log2 | about k | random order | full share | gain, nats |
|---|---|---|---|---|---|
| elicit map | 7.92 | 242 | 11.82 | 0.936 | 6.32 |
| cipher, same size | 8.88 | 473 | 12.37 | 0.694 | 10.53 |
| cipher, own step | 8.01 | 258 | 12.16 | 0.973 | 13.73 |
| same word back | 9.18 | 581 | 12.19 | 0.562 | 10.11 |
| random word, step 10 | 8.48 | 356 | 12.26 | 0.772 | 11.68 |
| same word, other keys, beside | 7.74 | 214 | 12.22 | 0.706 | 10.17 |
| random word, other keys, step 30, beside | 7.67 | 203 | 12.06 | 0.968 | 13.97 |
| lowercase | 3.86 | 15 | 11.97 | 0.712 | 1.57 |
| 500-word cipher | 6.76 | 109 | 12.27 | 0.519 | 3.27 |
| 50-word cipher, beside | 6.93 | 122 | 12.32 | 0.463 | 3.75 |
| control, 16 neurons | 1.99 | 4 | 13.03 | 0.163 | -1.53 |

## What we learned

Neurons do not map eliciting more cleanly, at this size and on these pairs, with "eliciting" meaning built to elicit. Weight updates are no use for this: they are flat after Adam and do not align with existing neurons. Activation shifts and per-window counts do differ between tasks, by more than seeds. But the direction differs between families, and each family compares very few tasks. Ranking per window shows that the gain sits on far fewer neurons per window than E56 suggested, for both kinds. The decomposition's share of the gain carried by changed CI values, $r$, separates family A at equal progress in E56, 0.76 against 0.25. It was not read on families B and C here, and it has no causal check yet either. So this is a comparison on one pair, and $r$ stays a candidate instrument under row 112.

## How compelling is the evidence

- **Controls:** all held, above. Recomputed rules agree with `score.json` on every pass, gap, reference gap and guard.
- **Blind to:** each number's line is in the table under "What was measured".
- **Noise floor:** seed ranges within a task, listed in the score table; on the count, one grid step by rule. Family A has five seeds, the others two.
- **Tasks covered:** three families, 26 states on the 67M model, Adam fine-tunes only. Not every task of either kind, and no causal labels.
- **Design limit:** one task against one task on three of the four tests (Amendment 15). The unscored tasks show how far tasks of one build sit apart: on the count, 1.44 between two same-word tables and 0.17 between two ciphers.
- **One number seen early:** the update's spread, as disclosed.

## Surprises

No filed prediction reversed. One missed narrowly: the count was filed to pass at matched size, and its gap of 0.965 fell 0.035 under the floor of 1, with its direction as filed.

Three things no prediction covered:

- The direction flips on family B for both the activation shift and the count. The cause is not established.
- With more training on the cipher the activation shift spreads while the neurons needed fall: 0.030 to 0.043, and 8.88 to 8.01. We did not test why.
- The first gradient is about as narrow for shuffled targets as for true ones. No value had been filed for this.

## What next

- **The truth-set reading needs the user's say,** as the tree states. If the activation shift and the count follow a causal label there, neurons return as a rival to $r$.
- **Re-read these numbers on any pair that gets a causal check.** It costs nothing on saved states. If direction then follows the label, family B's flip was about its label.
- **Read the elicit map at another step.** The progress reading rests on the cipher alone. A count that falls on elicit too would support it.

## Scorecard

Codes: W1 update spread, W1g first-gradient spread, W2 rescale share, A1 shift spread, C1 neurons carrying the gain. T1 matched size, T2 equal progress, T3_B family B, T3_C family C. "Gap" is signed so that elicit sparser is positive. "Over" is the spread it must strictly exceed. "Reference gap" is the same signed gap on each reference, which must be beaten.

| number | test | elicit | new | gap | over | reference gap | separates | elicit sparser | beats reference | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| W1 | T1 | 0.9761 | 0.9704 | -0.0056 | 0.0051 | rot -0.0598 | yes | no | yes | fail |
| W1 | T2 | 0.9757 | 0.9225 | -0.0533 | 0.0106 | rot -0.0154 | yes | no | no | fail |
| W1 | T3_B | 0.9744 | 0.9699 | -0.0045 | 0.0031 | rot -0.0051 | yes | no | yes | fail |
| W1 | T3_C | 0.9379 | 0.8162 | -0.1216 | 0.0033 | rot -0.0803 | yes | no | no | fail |
| W1g | T1 | 0.0530 | 0.0382 | -0.0148 | 0.0399 | shuf -0.0054 | no | no | no | fail |
| W1g | T2 | 0.0657 | 0.0389 | -0.0267 | 0.0277 | shuf -0.0107 | no | no | no | fail |
| W1g | T3_B | 0.0395 | 0.0372 | -0.0023 | 0.0053 | shuf -0.0035 | no | no | yes | fail |
| W1g | T3_C | 0.0152 | 0.0124 | -0.0028 | 0.0041 | shuf +0.0004 | no | no | no | fail |
| W2 | T1 | 0.00175 | 0.00196 | -0.00021 | 0.00008 | Gauss -0.0000006 | yes | no | no | fail |
| W2 | T2 | 0.00174 | 0.00143 | +0.00032 | 0.00008 | Gauss +0.00001 | yes | yes | yes | **pass** |
| W2 | T3_B | 0.00199 | 0.00199 | -0.00001 | 0.00012 | Gauss +0.0000015 | no | no | no | fail |
| W2 | T3_C | 0.00202 | 0.00191 | +0.00011 | 0.00002 | Gauss -0.00001 | yes | yes | yes | **pass** |
| A1 | T1 | 0.0246 | 0.0298 | +0.0051 | 0.0042 | Gauss -0.0003, rot -0.0087 | yes | yes | yes | **pass** |
| A1 | T2 | 0.0247 | 0.0430 | +0.0183 | 0.0002 | Gauss +0.0007, rot +0.0480 | yes | yes | no | fail |
| A1 | T3_B | 0.0375 | 0.0294 | -0.0081 | 0.0017 | Gauss +0.0048, rot -0.0085 | yes | no | no | fail |
| A1 | T3_C | 0.0123 | 0.0276 | +0.0152 | 0.0029 | Gauss +0.0024, rot -0.0155 | yes | yes | yes | **pass** |
| C1 | T1 | 7.92 | 8.88 | +0.965 | 1 | rnd +0.549 | no | yes | yes | fail |
| C1 | T2 | 7.84 | 8.01 | +0.166 | 1 | rnd +0.268 | no | yes | no | fail |
| C1 | T3_B | 9.18 | 8.48 | -0.704 | 1 | rnd +0.069 | no | no | no | fail |
| C1 | T3_C | 3.86 | 6.76 | +2.902 | 1 | rnd +0.301 | yes | yes | yes | **pass** |

| number | guard reference | elicit mean | reference mean | gap on the sparser side | range | verdict |
|---|---|---|---|---|---|---|
| W1 | rotated | 0.9761 | 0.9231 | -0.0530 | 0.0051 | fail, wrong side |
| W1g | shuffled targets | 0.0530 | 0.0433 | -0.0097 | 0.0399 | fail, wrong side |
| W2 | Gaussian | 0.00175 | 0.00131 | +0.00044 | 0.00008 | **pass** |
| A1 | Gaussian update | 0.0246 | 0.0525 | +0.0278 | 0.0042 | **pass** |
| A1 | rotated | 0.0246 | 0.1632 | +0.1386 | 0.0042 | **pass** |
| C1 | random order | 7.92 | 11.82 | +3.90 | 0.38 | **pass** |

**Passed by number.** W1 none. W1g none. W2: T2, T3_C, guard. A1: T1, T3_C, both guards. C1: T3_C, guard. None answers yes; `score.json` agrees.

| lean | filed | right if | read | verdict |
|---|---|---|---|---|
| P1 | W1 does not answer yes, Adam flattens it | W1 answers yes is false | false, fails all four tests and the guard | **right**, but seen before the amendments, not blind. That Adam is the reason was not tested |
| P2 | W1g unknown | no lean to score | fails all, elicit never the sparser side | reported |
| P3 | W2 does not answer yes | W2 answers yes is false | false; passes T2, T3_C and the guard, with gaps of at most 0.0003 on a scale where the control reads 1 | **right** |
| P4 | A1 unknown | no lean to score | passes T1, T3_C and both guards; T2 fails on the rotated gap, T3_B reverses | reported |
| P5 | C1 passes T1 and fails T2, as E56's coarse count did | C1 T1 passes and T2 fails | T1 fails: 0.965 under the floor of 1, Amendment 13. T2 fails: 0.166 | **missed**, narrowly. The T2 half and the direction held. The cause is arithmetic: the gap on the filed target is under the filed floor; the target's lower bar for the cipher contributes (on E56's target, unfiled, the gap is 2.36) |

## Departures from the literature

- **Neuron patching departs from causal tracing,** Meng et al. 2022. Causal tracing restores clean activations into a corrupted run of one model. We give the fine-tuned model's MLP activations to the original model on the same input, at every position. Neurons are ranked here per window by size of shift, without target information, and the reading is a share of the fine-tune's gain. Unit, ranking and the halves are this project's, from E52 and E56; the per-window ranking is new here.
- **The participation ratio** is the standard form, applied to neuron energies and shifts as a fraction of the neuron count. No paper's procedure was followed.
- **Rotated, shuffled-target, Gaussian-update and random-order references** are this project's. Rotation is not a symmetry of the model because of the GELU, so a rotated value is a comparison and not a null (Amendment 19).
- **E51's separation rule** is this project's, with two additions here: the one-sided gap must beat its reference's, and the count's spread has a floor of 1.
- **Thresholds have no external source:** the floor of 0.05 nats, the floor of 1, and the stage-0 tolerances.
- **The label "built to elicit"** departs from Jain et al. 2024, Def 1, which defines possession by a linear readout. It also departs from this project's mechanistic definition, row 112. No readout or causal check is used.

## Provenance

- Job 48527, MATS `debug` QoS, L40, 2 min 12 s, peak GPU memory 7.5 GB, $0. Log `cluster/logs/commute-neuron-sparsity-48527.out`.
- Results `experiments/neuron_sparsity/results/{predictions,stage0,weights,gradients,windows,score}.json`, committed in 1132c03. The JSONs record `git_head` de14090; the run's code files, predictions and sbatch are unchanged between it and 5727b38 (`git diff --stat` empty). `git_dirty` lists only `experiments/qwen_sft/results/run.log`.
- Code `experiments/neuron_sparsity/run.py`, `src/commute/neuron_sparsity.py`. Figures `experiments/neuron_sparsity/plot.py` into `experiments/neuron_sparsity/figures/`.
- E56 comparison curves `experiments/keydependent_patch/results/{job1,job2}.json`.
- Recomputation of the score, not committed: `~/commute-data/scratch/baseline-lead/e63-analysis/recheck.py`.

## Not run

- SGD fine-tunes of the pairs; the first gradient stands in.
- Attention-head sparsity.
- Any new fine-tune; Qwen.
- The causal label of `decisions.md` row 112, which E64 and E55 pursue.
- The truth-set reading of E55, which needs the user's say.
- The tree's follow-up on the equal-prior and shuffled maps, E58 A6-A7, which only a yes opens.
- The first-gradient reading on family C of the task's text minus the cased rows, the review's finding 7.
- The elicit map at any other step.
- The count of distinct neurons over all windows.
- $r$ on families B and C in this run; its values above are E56's, on family A.
- A1, C1 and W1g on the update that learned nothing: filed as cut, since its task has no answer position.
