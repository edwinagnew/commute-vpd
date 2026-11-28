# What the Qwen appendix says once its share is read as on the 67M (cell P2.90, 2026-11-27)

We read the Qwen3-1.7B fine-tunes of Appendix H again with the 67M model's share, built on the whole trained
answer's log-odds, beside random changes scaled to raise the answers by about one log-odds. On the VPD decomposition
the fine-tunes now read about 0.56 to 0.74, not 0.99, and random changes read lower on cipher, invented facts and
both maths tasks but as high on copy and known facts. So the appendix's saturation claim does not hold, and any
comparison of the Qwen fine-tunes with random changes holds only task by task.

## What was read

The share of an update is the mean over whole answers of the masked model's answer log-odds change when Γ's gates are
read on the fine-tune rather than the base model, over the mean change of the target model's answer log-odds
(`src/commute/share.py:5`, `answer_logodds` at :110, `cells` at :153). An answer is one whole trained response inside
one of the side's own 128 rows of 256 tokens, and its probability includes the end-of-response token
(`spans_from_marks` keeps the tokens through the end token, `share.py:81`); an answer cut by a row's edge is left out.
The whole answers read: copy 1,732, cipher 1,591, known facts 1,066, invented facts 901, maths 97 on 91 of 128 rows
(`results/run_qwen.log:1-5`).

- E327 (`experiments/share_unified/results/repro_67m.jsonl`) re-read the 67M paper's 48 re-gating states with this
  code on a 4090. On the same forwards each new value equals the old one-position unit exactly (all 67M answers are
  one token). Against the paper's values, 47 of 48 lie within the filed 0.002. The 48th, lowercase text seed 0, is a
  miss: it moves by 0.00245 (0.7519 to 0.7494), 0.00045 over the bar. None of the 48 matched the paper bit for bit,
  as the filed expectation for the same card and batch said they would, so this card or path differs from E136's;
  the 0.002 floor was measured across batch sizes on one card, not across cards. A card change is the likely cause
  of the miss but is unverified. The shares span 0.12 to 1.10, so it moves no reading. Per state:
  `results/known_share_check.csv`.
- E328 (`results/qwen_reread.jsonl`, 58 items) read the figure's 36 fine-tune states (6 sides, seeds 0-2, last save,
  two decompositions), its 12 run-1 random changes and 10 base-model rows on one H100. The output-distribution
  cells taken from the same logits match the saved ones within 0.052% relative, so the states and rows are the old
  ones. The base-model rows are a determinism check, not a control of the share: with the base model on both sides
  of every cell, on 16 rows (20 for maths), the numerator is exactly 0 on all 10 row sets, which any deterministic
  forward would give.
- The run-1 random changes were matched to the fine-tunes in output-distribution change, and 11 of the 12 lower the
  trained answer (negative gain; the twelfth raises it by 1.19 log-odds). They are not used as references here.
- E330 (`results/qwen_random_{vpd0,vpd1,neuron0,neuron1}.jsonl`, four H100 pods) drew the references the figure now
  uses, described in the next section.

New figure values (`experiments/share_unified/results/qwen_share_new.csv`, built by
`experiments/share_unified/build_csv.sh`; mean of three seeds, seed range):

| Side | VPD | One subcomponent per MLP neuron |
|---|---|---|
| Copy | 0.623 (0.611-0.637) | 0.441 (0.416-0.460) |
| Cipher | 0.572 (0.561-0.578) | 0.300 (0.296-0.304) |
| Known facts | 0.585 (0.576-0.597) | 0.303 (0.299-0.309) |
| Invented facts | 0.647 (0.642-0.651) | 0.351 (0.349-0.352) |
| Maths, real solutions | 0.737 (0.737-0.737) | 0.299 (0.298-0.301) |
| Maths, shuffled solutions | 0.709 (0.707-0.710) | 0.304 (0.302-0.306) |

Old values: 0.991-0.999 on VPD and 0.926-0.984 on the neuron control (the paper's `figures/data/qwen_share.csv`).
The old form also read 0.99 to 1.01 for VPD's random changes, so on VPD it never told a fine-tune from a random
change. Every fine-tune's gain is 10 to 160 log-odds, far above any floor. The sides of a pair are at their saved
sizes, not matched in update size, so no pair is compared here.

## Against random changes that raise the answers as much as one log-odds

E330 drew three random changes per side and decomposition, as the 67M paper's references are drawn. Each is Gaussian
noise at the per-matrix norms of one fine-tune seed, with draw seeds 228001 to 228003. Its sign is chosen by the odd
part of the gain, and one scale factor moves its gain into the band 1.003 to 1.097 log-odds, capped at 100 times.
The zero change reads gain and numerator exactly 0 on every row set, a determinism check.

The reference is a random change scaled to raise the trained answers by about 1 log-odds, against the fine-tunes'
10 to 160, at 0.8 to 31 times the fine-tunes' norms. So the comparison is not like for like. At a gain near 1 any
fixed error in the numerator weighs 10 to 160 times more as a share, and the code itself calls shares below a gain
of 1 unstable. A random change matched to the fine-tunes' own gain was out of reach: at the cap, one VPD copy draw
reached a gain of about minus 7.

Eight of the 36 draws never reached the band, so their share was not read; the table marks each "no band". Two are
on VPD, on cipher and invented facts. Six are on the neuron control, on cipher, invented facts and both maths tasks.
Their largest gains were 0.0002 to 0.84 log-odds. They cluster on the tasks built new, so leaving them out is
not neutral. The 28 read draws have gains of 1.010 to 1.093.

Every draw's share, "no band" for a draw that never reached the band
(`experiments/share_unified/results/qwen_share_random_draws.csv`, built by `build_csv.sh` from the jsonl):

| Side | VPD draws 0 / 1 / 2 | VPD fine-tunes | Neuron draws 0 / 1 / 2 | Neuron fine-tunes |
|---|---|---|---|---|
| Copy | 0.719 / 0.630 / 0.666 | 0.611-0.637 | 0.434 / 0.534 / 0.462 | 0.416-0.460 |
| Known facts | 0.687 / 0.515 / 1.031 | 0.576-0.597 | 0.373 / 0.229 / 0.726 | 0.299-0.309 |
| Cipher | no band / 0.375 / 0.351 | 0.561-0.578 | no band / no band / 0.086 | 0.296-0.304 |
| Invented facts | no band / 0.397 / 0.375 | 0.642-0.651 | no band / no band / −0.277 | 0.349-0.352 |
| Maths, real solutions | 0.554 / 0.553 / 0.483 | 0.737 | no band / −0.0004 / 0.055 | 0.298-0.301 |
| Maths, shuffled solutions | 0.551 / 0.538 / 0.478 | 0.707-0.710 | no band / −0.017 / 0.051 | 0.302-0.306 |

On VPD, every fine-tune seed reads above every read draw on four tasks. Cipher reads 0.56-0.58 against 0.35-0.37 on
two draws. Invented facts read 0.64-0.65 against 0.37-0.40 on two draws. Real-solution maths reads 0.74 against
0.48-0.55 on three draws, and shuffled-solution maths 0.71 against 0.48-0.55 on three. On copy the three draws read
0.63-0.72, as high as the fine-tunes' 0.61-0.64. On known facts two draws read 0.51 and 0.69 against the fine-tunes'
0.58-0.60.

The third known-facts draw reads 1.03 and is one draw, not the end of a measured range. Its numerator is 1.070 over
a gain of 1.037, and nothing bounds the ratio at 1. Draw to draw, known facts spread by 0.52 on VPD and 0.50 on the
neuron control, against a seed spread of 0.02 or less for the known-facts fine-tunes. So one known-facts draw alone can
land above or below the fine-tunes.

On the neuron control, every fine-tune seed reads 0.30 on both maths tasks, against draws at 0.055 or below, two draws
each. On copy the three draws read 0.43-0.53 against the fine-tunes' 0.42-0.46. On known facts they read 0.23-0.73
against 0.30-0.31. Cipher and invented facts rest on one draw each, at 0.09 and −0.28, against fine-tunes at 0.30 and 0.35.
So on copy and known facts, the two reuse tasks, a random change that raises the answers by one log-odds reads as
high as the fine-tunes on both decompositions.

## The paper statements that change

Paper clone `~/commute-agents/writeup-lead/paper/paper/` at commit 105c329. "Holds" means the sentence is still true
as written under the new share. The list covers every rendered sentence that cites the old Qwen values or the old
form (a grep for Qwen, saturation, 0.99 and the KL form over the files `main.tex` inputs), plus one file it does not
input.

0. `main.tex:49` (abstract, writeup-lead version) "on Qwen3-1.7B the re-gating share saturates". Does not hold, as 1.
1. `sec5_results.tex:10` "On Qwen3-1.7B the re-gating share saturates on VPD". New: VPD fine-tunes read 0.56-0.74.
   Does not hold.
2. `sec5_results.tex:85` "The re-gating share breaks down completely on VPD: it reads 0.99 or more for every fine-tune
   and for random changes that move the output as much, so it cannot tell them apart". New: fine-tunes 0.56-0.74 per
   seed. E330's gain-matched random draws read below the fine-tunes on cipher and invented facts, two draws each.
   They read below on both maths tasks too, three draws each, but as high on copy and known facts. Does not hold:
   the fine-tunes no longer read 0.99, and whether a random change reads apart from them depends on the task.
3. `sec5_results.tex:85` "With one subcomponent per MLP neuron it reads 0.92 to 0.99 for fine-tunes and 0.81 to 0.92
   for random changes, but no pair separates from random changes on every random draw". New: fine-tune means
   0.30-0.44, seeds 0.30-0.46. E330's gain-matched draws read −0.28 to 0.73, per task in the table above, with
   cipher and invented facts on one draw each. The draw test was read in the old form on E320's size-matched draws
   and is not re-read. The numbers do not hold; the pair clause is unchecked.
4. `appH_qwen_short.tex:17` caption title "On Qwen3-1.7B, the re-gating share saturates on VPD" and "cross: one
   random change of the same KL". Does not hold. The crosses become E330's draws matched in gain, one dot per draw.
   The caption names the eight draws that never reached the gain band.
5. `appH_qwen_short.tex:21-22` "Every fine-tune reads between 0.990 and 0.9993, and random changes of matched output
   change ... read 0.993 to 1.009". New: fine-tunes 0.561-0.737 on VPD; VPD's gain-matched draws 0.351-0.719, plus
   one known-facts draw at 1.031. Does not hold.
6. `appH_qwen_short.tex:23-26` the norm outside the subcomponents' span (0.97 to 1.00) and the planted changes (0.89
   against 0.06; "the share differs by less than 0.01" between often- and rarely-active plants). The span ratio is a
   weight-norm reading and holds. The planted-change share was read in the old form and is not re-read; as written it
   now describes a form the figure no longer uses.
7. `appH_qwen_short.tex:26-29` "The share here is the KL form ...; on the 67M too, this form reads random changes of
   matched output change as high as most fine-tunes: 0.97 to 0.99 ... against 0.89 to 1.00". The first clause no
   longer holds: the share is the whole-answer log-odds form, the one the 67M uses. The 67M comparison stays true of
   the old form but no longer explains the Qwen figure.
8. `appH_qwen_short.tex:31-40` the neuron-control paragraph (pairs against three random draws at matched size; "each
   fine-tune ... reads above all three of its random changes"). Read in the old form; not re-read here. Against
   E330's gain-matched draws the last sentence does not hold on copy or known facts. There draws read as high as the
   fine-tunes, and on cipher and invented facts only one draw each was read.
9. `appE_methods_detail.tex:24` "Figure qwen_share one draw per side, scaled to within 1.5 times the KL of its first
   seed". Changes to E330's reference, scaled to a gain of one log-odds as the 67M's (`appE:26-27`).
10. `appE_methods_detail.tex:116-117` "The score is the log-odds of the trained answer, and the share is read only
    when the gain is at least one log-odds". On Qwen the score is the whole multi-token answer's log-odds; E328 used
    the two-sided floor of 1e-6 that the 67M's E136 reads use. Every Qwen fine-tune gain is at least 10, so either
    floor scores all of them. Holds, with "whole answer" to add for Qwen.
11. `appE_methods_detail.tex:120` "A KL form, used on Qwen, is in Appendix". Does not hold: Qwen now uses the same form.
12. `appE_methods_detail.tex:184-190` "On Qwen the log-odds form could not be scored, so the share compares output
    distributions instead ... summed over every position up to the read position". Does not hold: the log-odds form
    is scored on every Qwen state, over whole answers rather than every position.
13. `sec5_baselines_scorecard_caveats.tex:61-62` "fine-tunes read shares of 0.990 to 0.9998, and random changes ...
    0.992 to 1.009" and `:66` "The Qwen share is a KL ratio ... because the log-odds form could not be scored there".
    This file is not input by `main.tex` (grep), so it does not render; if it is revived, neither sentence holds.

## Not run

- A test of the doubt from the premise check `~/commute-data/premise/P2.90.md`: nothing yet shows the share means the
  same at an answer gain of 1 as at 100.
- A repeat read of the 1.03 known-facts draw on fresh rows, to tell the draw's value from the read's noise.
- A measure of the numerator's noise at a gain of 1; the zero change reads exactly 0 and says nothing about it.
- A random change matched to the fine-tunes' own gain of 10 to 160; out of reach at the 100-times cap.
- The planted changes of `appH:23-26` and E320's three size-matched neuron-control draws in the new form.
- A numeric check of the multi-token answer log-odds against a direct float64 sum on Qwen logits (asked by the
  redteam review, `reports/reviews/2026-11-27_review_p290-share-unified.md`); E327 tests only one-token answers.
- The five-times save; only the last save was re-read, as the figure plots.
- The figure itself: `make_qwen_share.py` still reads the old files, and the paper directory is not edited here.
