# Plan: has Qwen3-1.7B-Base seen GSM8K's test questions? The prefix test

Apparatus, not a finding: no E-code. A branch of the confirmed plan `reports/plans/2026-10-27_plan_qwen-fresh-numbers.md`
("Verdict ambiguous or conflict: run the prefix test on Qwen3-1.7B-Base ... before deciding"); the fresh-number check
read "ambiguous" (job 39129). Code: `experiments/qwen_prefix_test/run.py`, `cluster/qwen_prefix_test.sbatch`;
predictions: `experiments/qwen_prefix_test/results/predictions.json`. Method from Wu et al.
([2507.10532](https://arxiv.org/abs/2507.10532)), as summarised in `reports/analyses/2026-10-27_analysis_qwen-gsm8k-rl-headroom.md` §2.

## The question, in plain words
On GSM8K test problems with new numbers, the unchanged Qwen3-1.7B lost about 4 points against the originals on
GSM-Plus, but nothing measurable on GSM-Symbolic. That fits two stories: the model remembers some test problems, or
the new numbers are harder. This run asks the base model, the one that would hold any memory from pretraining, to
finish each GSM8K test question from its first 60%, and counts how often it writes the rest word for word. GSM8K
train questions, which Qwen has very likely seen, show what memory looks like; GSM8K-style questions generated in 2026,
which it cannot have seen, show what guessing looks like. Restated in the Terms: no update, no decomposition; a W₀
reading of the base model that decides whether GSM8K test can be the held-out score for the Qwen line's first update
(card `decision-qwen-baseline`).

## Design check (`/design`)
- **Goal served:** `GOALS.md`'s second question, new capability or a strengthened one. The Qwen line needs a
  task where the update is known to strengthen, scored on a held-out set that recall cannot inflate. This run decides
  whether GSM8K test is such a set.
- **Nearest tried:** the fresh-number check (`experiments/qwen_fresh_numbers`, job 39129). It measured accuracy on new
  numbers and could not separate memory from harder numbers; this run measures recall of the question text, which
  harder numbers cannot fake. No `TRIED.md` row covers either; the fresh-number check's row is being written now.
- **Patch or step:** a step: the second half of a two-part check, named in advance in the confirmed decision tree.
- **Decisions relied on:** none from `decisions.md` is touched (no answers are generated, so the 512-token window
  does not apply). No departure.
- **What changes next steps:** the verdict picks among the card's options: keep GSM8K test as the held-out score,
  make the fresh sets the primary score, or put a task switch to the user (decision tree below).
- **Control first:** a copy control runs first and alone. The model sees each question whole, then its own prefix,
  so it only has to copy; it must reach 0.90, else the job stops. It checks the cut, the batching, the decoding and
  the comparison. The memory control (GSM8K train) runs next. Nothing reproduces a reference procedure.
- **Approved class:** 3, a step of a confirmed plan's decision tree. Three choices below are mine, inside that step: the
  2026 synthetic set as the floor, the copy control, and cutting at a word boundary. The lead should glance at the first.

## Sets
| set | items | role | revision |
|---|---|---|---|
| GSM8K test, first 256 | 256 | copy control (stage 1) | `openai/gsm8k` 740312a |
| GSM8K train, first 1,319 | 1,319 | positive control: likely seen | same |
| Multilingual GSM-Symbolic, English `test_synthetic` | 2,000, 20 per template | floor: cannot have been seen | `danish-foundation-models/multilingual-gsm-symbolic` 75ce20d (tag v0.5.5) |
| GSM8K test, all | 1,319 | the reading | `openai/gsm8k` 740312a |
| model | | `Qwen/Qwen3-1.7B-Base`, bf16 | ea980cb |

**The floor.** The synthetic set renders the 100 GSM-Symbolic templates with new names and numbers. Each template is
one GSM8K test question; the 100 source ids equal GSM-Symbolic's (CPU check, 2026-10-27). The dataset was first
published on 2026-04-22, a year after Qwen3's release, so no variant can be in its training data. The headroom report
asked for fresh instances as the floor; Apple never released its generator, and this set is the same thing made by
a third party. GSM-Plus is not used as a floor: it was public from 2024, and each variant keeps its original's
wording except one number, so memory of the original would complete it too.

**The cut.** Each question is cut at a word boundary. The prefix is its first int(0.6 n) of n words, so no number or
word is split. The model gets the prefix as raw text: no chat template, no instruction. A match counts when the
continuation, with whitespace collapsed, equals the remainder, or starts with the remainder and then a space. Questions
average 45 to 47 words on all four sets; remainders average 24 tokens (longest 77).

## Stage 1: copy control (runs first, alone)
| number | must read | expected | else |
|---|---|---|---|
| EM_copy, 256 GSM8K test questions shown whole, then their own prefix | ≥ 0.90 | about 0.98 | stop; nothing else is read |

## Stages 2 to 4: train, floor, test; readings at the end
| number | how measured | expected | noise floor |
|---|---|---|---|
| EM_train | share of 1,319 train questions completed word for word | about 0.10 (0.02 to 0.40) | binomial SE about 0.008 |
| EM_floor | share of the 2,000 unseen variants completed word for word | about 0.02 (0.00 to 0.08) | SE over the 100 templates, about 0.005 |
| EM_test | share of 1,319 test questions completed word for word | about 0.05 (0.01 to 0.20) | about 0.006 |
| D_train = EM_train − EM_floor | the positive control | about 0.08; passes at ≥ 0.05 | about 0.010 |
| D_test = EM_test − EM_floor | test's excess over guessing | about 0.03 | about 0.008 |
| r = D_test / D_train | test's excess as a share of train's | about 0.35 | about 0.1 at D_train 0.08; 0.2 at 0.05 |

There is no do-nothing reference: no update is read. Expected values come from Wu et al.: Qwen3-4B-Base and 8B-Base
completed 21.8% and 32.0% of MATH-500 word for word from 60%, and 0% of AIME 2025, released after them. A 1.7B
model should reproduce less of what it saw.

**Reading r.** r near 0 means test questions are reproduced no more often than unseen ones; r near 1 means as often as
train's. If train was fully seen and recall per seen question is the same on both sets, r is the share of test seen as
often as train, and an upper bound otherwise. Memory could explain all of the 0.039 GSM-Plus gap only if roughly a sixth
of test were recalled (0.039 over the 0.26 the model gets wrong on new numbers). So r ≤ 0.2 leaves memory a minor part
of that gap; r ≥ 0.5 puts test with train. The cuts are coarse because r's noise is about 0.1.

**Blind to.** Memory that is not word for word: a problem learned from rephrased copies is missed, on test and on
train alike. Answer memory without question memory. The post-trained model, whose score is the one at stake: Wu et
al. found post-training lowers word-for-word completion (Qwen3-4B 5.2% against 21.8% for its base); the base is
where pretraining memory shows. Template memory also raises the floor: 7.8% of the floor's remainders are word for word
their original's (CPU check), so recall of an original can complete them. This shrinks D_test, by at most 7.8% of
test's recall rate.

**Positive control.** GSM8K train for memory detection, read as D_train ≥ 0.05. It may fail. CoDeC read Qwen2.5-1.5B-
Instruct as having seen GSM8K train (100 on its scale), but read Qwen3-8B and 14B at only 28 and 54, below its "seen" line
of 80. A failed control leaves a null on test unreadable; a high D_test still reads.

## Predictions (filed in `results/predictions.json`)
- P0: EM_copy ≥ 0.90. *Lean: high.*
- P1: D_train ≥ 0.05, the positive control passes. *Lean: medium.*
- P2: the verdict is not "like train". *Lean: medium.*
- P3: the verdict is "rare" (control passes and r ≤ 0.2). Predicted to fail, into "partial". *Lean: low.* Qwen3
  bases reproduce MATH-500 test questions, so their decontamination missed at least one well-known test set.

**Verdict rule.** If D_train < 0.05: "like train" when D_test ≥ 0.05, else "no control". If D_train ≥ 0.05: "rare" when r ≤ 0.2,
"like train" when r ≥ 0.5, "partial" between. Both cuts are inclusive; negative r reads "rare"; any missing set reads
"no reading". Edge cases run on CPU before filing: an empty set, D_train zero or negative (no division), exact ties
at 0.05, 0.2 and 0.5, r above 1, a floor with one template (SE None, verdict still read). The match rule accepts each
set's own remainders (1.000 on all four sets) and rejects a remainder whose last word runs on ("…b5" against "…b").

## Decision tree (card `decision-qwen-baseline`)
- P0 fails: stop; read the copy continuations and fix the apparatus. Nothing is read.
- **rare:** option (a) as recommended: keep GSM8K. GSM8K test stays the held-out score, with GSM-Plus and the 2026
  synthetic set beside it. The 0.039 GSM-Plus gap reads as harder numbers.
- **partial:** option (a), with the fresh sets as the primary held-out score: the 2,000 unseen synthetic variants and
  GSM-Plus. GSM8K test is reported beside them. Recommended on the card.
- **like train:** a card for the user, no default: (a) fresh-set scoring only, with GSM8K test dropped as a held-out
  score, or (b) a task switch, synthetic multi-step arithmetic first (headroom report §3).
- **no control:** the test cannot see memory at this size and settles nothing. Recommend option (a) with fresh-set
  scoring as primary, the choice that holds either way. Card to the user.

## Cost
One L40 on the cluster's debug QoS, about 10 minutes. The 3.4 GB base model downloads into `/ephemeral` (cluster/env.sh),
then 40 batches of 128 completions of at most 85 new tokens run. Each stage has a time budget (120, 300, 420, 300 s)
and stops early, recording k, rather than overrun 30 minutes. Peak GPU memory well under the 14 GB of the 512-token
runs. No paid compute.

## Not run
- Wu et al.'s second measure: whether the model gives the right final answer from the 60% prefix alone.
- The 40% and 80% prefixes; a partial-match score (ROUGE-L).
- The post-trained Qwen3-1.7B, whose score is the one at stake.
- The other 6,154 GSM8K train questions.
- A floor without the 7.8% of variants whose remainder equals their original's.
- The fresh-number accuracy gap of the post-trained model on the 2026 synthetic set (unseen, unlike GSM-Symbolic's
  2024 release that stage 3 of the fresh-number check used).
