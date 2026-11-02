# Analysis: does a readout trained only on Pile text find the half-known values before the last block? (E59, closed unreadable)

*2026-11-02, coding agent, baseline-lead E59 writer. Plan: `reports/plans/2026-10-30_plan_baseline-next2-draft.md`,
approved by the user, with its four dated amendments. Rules: `experiments/jain_readout/results/predictions.json`
(blocks `amended_2026_11_02_pre_run`, `amended_2026_11_02_after_stop`, `amended_2026_11_02_step250`,
`amended_2026_11_02_after_stop2`). Results: `experiments/jain_readout/results/run.json` (job 48412) and its git
history. Code review: `reports/reviews/2026-11-02_review_e59-code.md`. Advice adopted at the stops:
`reports/reviews/2026-11-02_review_e59-stop-fable-advice.md`, `reports/reviews/2026-11-02_review_e59-stop2-fable-advice.md`.
Explanation: `notebooks/E59_jain_readout.ipynb`. Provenance at the end.*

We trained one linear readout per block on Pile text, to see whether a simple probe finds the half-known values inside the original model. The readouts memorised their training text and missed the model's own ranks at the last block, so the test failed its construction check unanswered. So the label "strengthens what the model half-knows" still rests on the model's own output, and no label for the pair yet avoids both that output and the decomposition.

## The answer

We could not tell. A next-token readout over the full vocabulary, trained on the token budget the free queue allows, is too weak on this model. It never held the elicit values at any block, and at the last block it disagreed with the model it was meant to copy. No block held the cipher's values either, but a readout this weak would miss them anyway, so that agreement adds little. The removal tests on the fine-tunes never ran.

## The question, and why this test

E54 built a pair of fine-tunes on the 67M model over the same 32 keys. Each window is Pile text with a rare marker, then a key; the model must predict the key's value next.

- **Elicit map.** Each value is a word the original model's averaged prediction after the key already ranks 2nd to 20th.
- **Cipher.** The same keys mapped to words the original model almost never predicts there.

The decomposition separates them. Let only Γ, the causal-importance function, see the fine-tuned activations, with every subcomponent's weights kept at their original values. That recovers about three quarters of the elicit map's log-odds gain and about a quarter of the cipher's, on five seeds in E56. The objection is that "half-known" was read off the original model's own output ranks, so the pair may have been built to make the decomposition read high. E59 asked for a label from outside both. It used Jain et al. 2024, Definition 1: a model possesses a capability when a linear readout trained on pretraining data recovers it from the model's activations (`decisions.md` row 81).

## What was measured

- **Readout.** One `nn.Linear(768, 50277)` per block, reading that block's output residual: blocks 0 to 2 at the next block's input, block 3 before the final norm. Trained on next-token loss over random 41-token windows of Pile train rows, positions 1 to 39, never on the task windows. Schedule from Jain et al. as copied in `src/commute/probe.py`: 4,000 AdamW steps at learning rate 1e-3, divided by ten at steps 2,000, 3,000 and 3,500, batch 96.
- **Rank of a value.** At the answer position of E54's 97 held-out task windows, one plus the number of words the readout scores strictly higher. A key's rank is the lower median over its windows.
- **Possessed at a block.** At least 24 of the 32 keys at median rank 20 or better, the band the elicit pair was built with.
- **Construction check.** At the last block, at least 24 elicit keys at rank 2 to 20 and at least 24 cipher keys above rank 100. A stop.
- **Readout quality, as amended after the first stop.** Last-block held-out loss within 0.1 nats of the linearised head, both on 2,000 windows of Pile validation rows. The linearised head is the model's own unembedding with the final RMSNorm's per-token scale fixed at its training-batch mean. It is a linear map the readout could learn, so it bounds the best linear readout's loss from above.
- **Removal share, never reached.** At the earliest block before the last that possesses the elicit map, remove the target value's readout direction from the residual at the answer position. Read the share of each saved fine-tune's log-odds gain this costs, net of what the original model loses under the same removal.

What the numbers read if the apparatus works: at the last block, a readout close to the model's head gives most elicit values rank 2 to 20. The model's averaged output does so for all 32. Every cipher value should sit far above rank 100.

## Expected

- The elicit map possessed at block 0, 1 or 2, the cipher at no block. Lean medium. If the elicit values were held only at the last block, "half-known" would mean "what the output already says".
- Removing the readout direction costs the elicit fine-tunes more of their gain than the matched ciphers, by E51's separation rule. Lean medium.
- The last block reproduces the construction. Lean high.

## Did the readouts learn the Pile, or memorise it?

![](../../experiments/jain_readout/figures/readout_loss.png)

**Reading it.** They memorised. On unseen Pile windows the last block's readout reached 3.94 nats, 0.37 above the linearised head, against a bar of 0.1. On windows drawn from its training rows it sat 0.07 above the head, inside the bar. The model and the linearised head read the same on both pools, so the whole gap belongs to the readout. Unseen text reads 0.22 to 0.30 nats worse than seen text at every block.

The cause, from the second advice report: each readout has 50,277 rows of 768 weights. A row learns where its word belongs mainly from tokens whose target is that word, so the rows of rare words see only a handful in 15 million tokens and fit them. Jain et al. read a vocabulary of a few hundred, where the schedule never meets this. More steps on fresh rows would roughly halve the gap per doubling of tokens, so closing it needs four to eight times the tokens.

Held-out Pile loss per block, nats, job 48357's readouts; job 48412 read them again with identical results:

| | block 0 | block 1 | block 2 | block 3 | model | linearised head |
|---|---|---|---|---|---|---|
| held-out, 2,000 val windows | 5.574 | 5.093 | 4.380 | 3.941 | 3.428 | 3.571 |
| held-in, 2,000 train-row windows | 5.358 | 4.859 | 4.127 | 3.646 | 3.428 | 3.572 |
| held-in minus held-out | −0.215 | −0.234 | −0.253 | −0.296 | 0.000 | 0.001 |
| job 48279, held-out, reused rows | 5.672 | 5.213 | 4.527 | 4.140 | 3.428 | not computed |

Derived: last block over the linearised head 0.371 held-out and 0.074 held-in; last block over the model 0.513; linearised head over the model 0.143; mean norm scale 1.1008.

### Four jobs, four stops

All on the MATS debug queue, L40, $0. Stage 0's input checks passed identically in every job: the twelve saved states' sha256 and gains, and the original model's rates, reproduce E54 and E56 with difference 0.

1. **Job 48279: overfit on reused rows.** The readouts drew 15.7 million window tokens from 4.2 million Pile tokens, so each token was fitted about 3.7 times. The last readout's training-batch loss settled near 3.2 from step 2,000, below the model's 3.43 on unseen text, while its own unseen loss was 4.14. It missed the first-filed bar of 0.5 nats over the model by 0.21. Amendment: train on rows 32768 to 65535, so each token is seen less than once in expectation. The guessed bar becomes the linearised head plus 0.1, with early checks added and the readouts saved.
2. **Job 48344: stopped at the step-250 early check.** The last block's batch loss read 5.0088 against a bar of 5.0. That bar came from one batch of job 48279 reading 4.73. Batches vary by about 0.14, so the bar sat within two standard deviations of a normal reading. Amendment: the step-250 bar raised to 6.0, still far below the untrained 10.9, and the step-2,000 bar to 1.0 over the model. The cache job 48321 ran on CPU before it.
3. **Job 48357: trained cleanly, missed the quality bar.** As in the figure and table above. The seen-versus-unseen comparison showed a sample-size limit, not a bug and not under-training. Amendment, on the second advice report: run the saved readouts through the rest of stage 0 and the predictions. The loss rule is recorded rather than a stop, and a reading rule is fixed first. A pass stands, since a weak readout biases against the claim. An elicit-side miss on possession, or a removal-share failure, reads "unreadable". A cipher-side miss is real. The construction check, the do-nothing and the original model's control stay stops. Hard cap: this one run.
4. **Job 48412: the saved readouts read the task windows.** Possession was scored, then the construction check failed and the run stopped before any removal.

## Does any block's readout hold the elicit values, and none the cipher's?

![](../../experiments/jain_readout/figures/possession.png)

**Reading it.** No block holds the elicit map, and the last block fails the construction check, so the readout cannot answer the question. Block 0 comes closest, with 21 of 32 keys at rank 20 or better against the 24 needed. At the last block 11 elicit keys sit at rank 2 to 20, where the model's averaged output puts all 32 by construction. No cipher value reaches rank 20 at any block; the typical cipher rank is several thousand.

Keys at median rank 20 or better, all 97 held-out windows, and on the 49-window fit half used to choose the removal block:

| block | elicit, all | elicit, fit half | cipher, all | elicit at rank 2-20 | elicit, middle key's rank | cipher, middle key's rank | cipher, best key's rank |
|---|---|---|---|---|---|---|---|
| 0 | 21 / 32 | 17 / 28 | 0 / 32 | 17 | 13 | 4,585 | 615 |
| 1 | 17 / 32 | 15 / 28 | 0 / 32 | 16 | 16 | 4,362 | 929 |
| 2 | 17 / 32 | 13 / 28 | 0 / 32 | 16 | 20 | 5,488 | 1,375 |
| 3 (last) | 13 / 32 | 13 / 28 | 0 / 32 | 11 | 25 | 6,849 | 800 |

The fit half covers 28 keys, so possession there needs 21. The middle key is the 17th of 32 in rank order. At the last block the elicit ranks run from 1 to 14,060. Two keys sit at rank 1, just outside the construction band. Several sit in the thousands, where the model ranks every elicit value in its top 20.

**How much weight the cipher clause bears: little.** The pre-set rule says a cipher-side *fail* would be real, because a weaker readout cannot push a value up. The converse does not hold. A weaker readout ranks every value worse, which pushes towards "not held", the very outcome the cipher clause needed. And the cipher values were chosen as words the original model almost never predicts after the key, so any readout that tracks the model even roughly ranks them low. The readout does separate the two value sets at every block, tens against thousands. That agrees with the label, but it is the same fact the label was built from, read through a worse instrument.

**One unexplained feature.** The elicit count falls with depth, 21 keys at block 0 to 13 at the last, and the seen-versus-unseen gap grows with depth. A readout that tracked the model well would hold the most elicit values near the output. We did not test why. One guess is that the block-0 readout acts like a table of likely next words after the key, which is how the elicit values were picked.

## What we learned

The question stays open. A full-vocabulary next-token readout on 15 million Pile tokens neither matches the 67M model's head on unseen text nor reproduces its ranks for the elicit values. So it cannot say whether an earlier block holds them. For `GOALS.md`'s question of whether the decomposition reads "teaches new" against "strengthens existing", the label "existing" for this pair still rests on the original model's own output ranks. We have no label for it that is independent of both VPD and that output. The decomposition's separation of the pair stands as measured in E54 and E56, but what it separates is still defined by the output.

## How compelling is the evidence

- **For "unreadable": strong.** The construction check missed by 13 keys, and the loss gap has a cause shown by the seen-versus-unseen comparison.
- **For anything about the model: weak.** The readout is blind to non-linear encodings and to directions the model does not read. It also ranks among 50,277 words, where the paper ranks among a few hundred.
- **Noise.** About three held-out windows per key, so a key's median rank is noisy; the 24-key count absorbs some of that. One key draw, one readout seed, one model.

## Surprises

- The elicit count falls with depth and is lowest at the last block. Unexplained.
- Fresh data removed the reuse but not the memorisation: vocabulary size, not repetition, was the limit. Neither advice report foresaw it before job 48357.

## What next

- **A small-vocabulary readout,** restricted to the table's values as in Jain et al. Each row then sees many examples. If it holds the elicit values at an early block and not the cipher's, the label becomes independent of the output. If it holds them only at the last block, "half-known" means "what the output says" and the write-ups say so. A new readout definition, so it goes through `/design` and the user.
- **The same readout on four to eight times the tokens.** One to two hours of an L40 at the measured 0.23 seconds per step. The normal queue is for the Qwen line only, so this is paid or needs the user.
- **The graded ladder,** the plan's option b: tables over the same keys at several prior-rank bands. It keeps a behavioural label but would show whether the decomposition's reading rises with how well the model knows the value. The plan's decision tree does not take its removal-failure branch on an unreadable result, so this is a fresh ask.

## Scorecard

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| stage 0 inputs | saved states and rates reproduce E54, E56 | sha256 equal; gains and rates within 1e-6 | all twelve states exact, difference 0 | pass |
| stage 0 readout loss | last block fits the Pile | held-out over the linearised head ≤ 0.1 | 0.371; held-in 0.074 | miss; recorded, not a stop, under `amended_2026_11_02_after_stop2` |
| P3 | last block reproduces the construction | ≥ 24 elicit keys at rank 2-20 and ≥ 24 cipher keys above 100 | elicit 11, cipher 32 | **fail**; the run stopped |
| P1, elicit clause | elicit possessed before the last block | some block 0-2 with ≥ 24 of 32 keys at rank ≤ 20 | best block 0, 21; none possessed | **unreadable** by the pre-set rule |
| P1, cipher clause | cipher possessed at no block | no block 0-3 with ≥ 24 keys at rank ≤ 20 | 0 keys at every block | holds; little weight |
| P2 | the gain runs through the readout direction | E51's rule on log-odds removal share, gap ≥ 0.2 | — | not run |
| do-nothing | random direction costs nothing | removal share within 0.05 on every state | — | not run |
| W0 control | the original model uses the direction | median drop ≥ 0.3 nats at the removal block | — | not run; no removal block existed |

Reading per the rule fixed before job 48412: the elicit clause is unreadable and the removal test did not run. The construction check's failure means the readout does not reproduce the model's own ranks. So the apparatus is not valid for this question at this budget, and per the hard cap E59 is closed.

## Departures from the literature

- **Vocabulary.** Jain et al. read a vocabulary of a few hundred; this readout covers all 50,277 tokens. That is what made it under-fit, and it was named only after job 48357.
- **Training data.** The paper trains the readout on samples from the pretraining process, fresh each step. We train on Pile next-token windows; job 48279 reused its rows about 3.7 times, and from job 48357 on each token was seen less than once in expectation.
- **Position.** The paper reads the readout on the sub-domain of inputs carrying the task. We train at every position from 1 to 39 of a window, position 0 left out for its attention-sink norm, and read only at the answer position after the marker and key.
- **Thresholds.** The paper gives no number for "recovers the capability". Ours, median rank 20 or better for 24 of 32 keys, and the construction band of rank 2 to 20, were filed before the run and follow the pair's construction. The linearised-head bar of 0.1 nats, the early-check bars, and the reading rule after the third stop are ours.
- **Causal check.** The paper prunes neurons or fine-tunes in reverse; the removal of the readout direction is `decisions.md` row 81's addition. Neither ran.
- **Schedule** is the paper's, as copied in `probe.py`; batch 96 and AdamW defaults were filled in there.

## Not run

- The removal tests on the twelve saved fine-tunes: the scored removal share, the random-direction do-nothing, the original model's control, and the last-block removal.
- Readouts on four to eight times the tokens, which need the normal queue or money.
- A small-vocabulary readout restricted to the table's values, which would follow Jain et al. more closely.
- Reverse fine-tuning, the paper's own causal tool.
- The own-step cipher at seeds 2 to 4, never saved by E56; the plan's fifteen states were twelve.
- Another key draw, another readout seed, a per-key breakdown, or any of this on Qwen.

## Provenance

- MATS cluster, debug queue, L40, $0. Jobs 48279 (16 min of training), 48344 (stopped after 1 min), 48357 (16 min), 48412 (under 1 min; reloaded 48357's readouts); CPU cache job 48321. Logs `cluster/logs/commute-jain-readout-{48279,48344,48357,48412}.out`.
- `experiments/jain_readout/results/run.json` at commit 25ba4fe, job 48412. Earlier versions: 4851105 (job 48279), 2f167bf (job 48344), 630cc63 (job 48357).
- Readouts `experiments/jain_readout/results/large/readouts.pt`, sha256 `374de867d10dda706075c359d001ae697678d9693ee13162179d5028b4e079d9`; job 48279's kept as `readouts_48279.pt`.
- Figures `experiments/jain_readout/figures/{readout_loss,possession}.png` from `experiments/jain_readout/plot.py`.
