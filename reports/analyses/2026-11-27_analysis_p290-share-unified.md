# What the Qwen appendix says once its share is read as on the 67M (cell P2.90, 2026-11-27)

We read the Qwen3-1.7B fine-tunes of Appendix H again with the same share the 67M model uses, the change in the
whole trained answer's log-odds, in place of the output-distribution form the appendix used. On the VPD decomposition
the fine-tunes now read between about 0.56 and 0.74, not 0.99 or more, and on the control with one subcomponent per
MLP neuron between about 0.30 and 0.46. So the appendix's claim that the share saturates on Qwen does not hold in this
form, and the random references the figure needs are still being drawn.

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
  of the miss but is unverified. The shares span 0.4 to 1.1, so it moves no reading. Per state:
  `results/known_share_check.csv`.
- E328 (`results/qwen_reread.jsonl`, 58 items) read the figure's 36 fine-tune states (6 sides, seeds 0-2, last save,
  two decompositions), its 12 run-1 random changes and 10 base-model rows on one H100. The output-distribution
  cells taken from the same logits match the saved ones within 0.06% relative, so the states and rows are the old
  ones. The base-model rows are a determinism check, not a control of the share: with the base model on both sides
  of every cell, on 16 rows (20 for maths), the numerator is exactly 0 on all 10 row sets, which any deterministic
  forward would give.
- The run-1 random changes were matched to the fine-tunes in output-distribution change, and 11 of the 12 lower the
  trained answer (negative gain; the twelfth raises it by 1.19 log-odds). They are not used as references here.
  E330's random changes matched to a gain of one log-odds will fill the figure's grey crosses; it is still running.

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
sizes, not matched in update size, so no pair is compared here. Nothing in this report compares the fine-tunes with
random changes, or VPD with the neuron control on that ground: that waits for E330.

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
   seed; the random changes are pending E330. Does not hold for the fine-tunes; the random half cannot be judged yet.
3. `sec5_results.tex:85` "With one subcomponent per MLP neuron it reads 0.92 to 0.99 for fine-tunes and 0.81 to 0.92
   for random changes, but no pair separates from random changes on every random draw". New: fine-tune means
   0.30-0.44 (seeds 0.30-0.46); random values pending E330; the draw test was read in the old form (E320's
   size-matched draws) and is not re-read. The numbers do not hold; the pair clause is unchecked.
4. `appH_qwen_short.tex:17` caption title "On Qwen3-1.7B, the re-gating share saturates on VPD" and "cross: one
   random change of the same KL". Does not hold; the crosses become E330's draws matched in gain.
5. `appH_qwen_short.tex:21-22` "Every fine-tune reads between 0.990 and 0.9993, and random changes of matched output
   change ... read 0.993 to 1.009". New: fine-tunes 0.561-0.737 on VPD; randoms pending. Does not hold.
6. `appH_qwen_short.tex:23-26` the norm outside the subcomponents' span (0.97 to 1.00) and the planted changes (0.89
   against 0.06; "the share differs by less than 0.01" between often- and rarely-active plants). The span ratio is a
   weight-norm reading and holds. The planted-change share was read in the old form and is not re-read; as written it
   now describes a form the figure no longer uses.
7. `appH_qwen_short.tex:26-29` "The share here is the KL form ...; on the 67M too, this form reads random changes of
   matched output change as high as most fine-tunes: 0.97 to 0.99 ... against 0.89 to 1.00". The first clause no
   longer holds: the share is the whole-answer log-odds form, the one the 67M uses. The 67M comparison stays true of
   the old form but no longer explains the Qwen figure.
8. `appH_qwen_short.tex:31-40` the neuron-control paragraph (pairs against three random draws at matched size; "each
   fine-tune ... reads above all three of its random changes"). Read in the old form; not re-read here. Unchecked;
   the last sentence waits on E330.
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

- E330's random changes matched in gain, the references for the figure and for statements 2-5 and 8; running.
- The planted changes of `appH:23-26` and E320's three size-matched neuron-control draws in the new form.
- A numeric check of the multi-token answer log-odds against a direct float64 sum on Qwen logits (asked by the
  redteam review, `reports/reviews/2026-11-27_review_p290-share-unified.md`); E327 tests only one-token answers.
- The five-times save; only the last save was re-read, as the figure plots.
- The figure itself: `make_qwen_share.py` still reads the old files, and the paper directory is not edited here.
