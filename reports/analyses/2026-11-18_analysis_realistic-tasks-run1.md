# Realistic tasks on the 67M, run 1: which real-text tasks can carry the comparison

We checked seven real-text fine-tuning tasks on the original 67M model before running any method on them: whether the
base model already holds what the reuse side needs, lacks what the new side needs, and whether each task can be learned
at the update sizes the built pairs use. Five of the six pairs' members pass and are kept, the sentiment task drops,
and one check fails in a way that changes how the label tasks must be read: most of their gain is the model learning to
say a label word at all, the same on both sides. For `GOALS.md` this means the real-text pairs exist at the right sizes,
but the topic pair can only test reuse against new if its gain is read between the two label words.

## What ran

One cluster job (E209, cell P2.38 run 1, job 61424, about 10 min on an L40, $0). Plan
`reports/plans/2026-11-18_plan_p2-38-run1.md`; rules `experiments/realistic_tasks/results/predictions.json`; results
`experiments/realistic_tasks/results/stage{0,1,2}.json`; log `cluster/logs/commute-realistic-r1-61424.out`.

**Control first.** Stage 0 re-read two saved E53 fine-tunes (lowercase and cipher) through the same readers. Every
number matched its saved value: share 0.7519 and 0.1243, gain 1.6721 and 3.8681. The zero update read exactly 0.
The apparatus reads as before.

## What the base model holds (stage 1)

| prediction | bar | read | holds |
|---|---|---|---|
| P1 topic probe (L1, Sports vs Business) | at least 0.9 | 0.969 | yes |
| P2 coin probe (L3, random label) | at most 0.6 | 0.506 | yes |
| P3 sentiment probe (L2, SST-2) | at least 0.7 | 0.658 | **no**: L2 dropped |
| P4 absent language against French, nats per byte | ratio at least 1.2 | Yoruba 2.15 against 1.21, ratio 1.78 | yes |
| P5 reversed word order costs nats per token | at least 1 | 5.42 against 3.14 in order | yes |

The sentiment miss was the one rated medium: the base model reads sentiment linearly only at 0.66 at its best block.

## Can each task be learned at the built pairs' sizes (stage 2)

Every kept task passes its gain bar at learning rate 3e-5 (P6), at an update size of 0.021 or less. T1 reproduced
E53's recipe: gain 0.1295 at size 0.0179 (filed 0.12963 within 0.003). The language tasks: French gains 0.17 nats per
token, Yoruba 0.90, reversed order 1.11, each at step 100.

**The miss.** The plan expected the coin task's gain on held-out texts to be near 0, since a coin cannot be predicted.
It reads 8.96 log-odds. The gain is the log-odds of the right label word against the whole vocabulary. A coin's held-out gain can only come
from raising both label words over the rest of the vocabulary, so about 9 log-odds is learning to output a label word
at all. The topic task gains 17.6 on held-out texts, so about half of it is that same common part. Read this way, the pair's two sides share a large common part, and a method could separate them or not
because of that common part.

**What changes for run 2.** The label pair's gain and every reading built on it are taken on the two-label contrast:
log p(right label) minus log p(other label). Then the coin task's held-out gain should read near 0 and the topic
task's should not. The whole-vocabulary reading is reported beside it. This is a measurement choice inside the cell
(user, 2026-11-17: measurement details are the lead's call), recorded in `decisions.md`.

## Not run

- No method (share, steering, neurons, crosscoder) on the new tasks yet: that is run 2 and run 3.
- Seeds 1 and 2; training each pair to equal size.
- The two-label contrast gain itself: it was not saved in run 1.
- FLORES+ parallel sentences: the dataset is gated, so the languages are compared on different Wikipedia articles.
- Qwen.
