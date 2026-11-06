# E112: does a case-label fine-tune of the 67M model reuse "the previous word is capitalised"?

*definition-lead, 2026-11-06. Plan `reports/plans/2026-11-06_plan_e112-case-label.md`; filed predictions
`experiments/case_label/results/predictions.json`; notebook `notebooks/E112_case_label.ipynb`. Provisional: no
independent check.*

We taught the small model to pick a label word by whether the previous word is capitalised, beside a twin that learns
labels from a shuffled list. The real fine-tune looks like reuse of the model's own case feature on unseen words, but
removing that feature's one direction took away under a tenth of its gain. Under the adopted definition the real task
therefore reads "not through the capability", our filed prediction missed, and this pair gives no reuse label yet.

## The answer

The real fine-tune passes the definition's first two tests and fails its causal check. The original model holds the
case of the previous word at its second block. The fine-tuned model labels unseen words almost perfectly, and its
labels match a single cut on that case readout. But removing the readout's one direction at that block and position
took only 8 to 9 percent of the label gain, against a filed bar of one half. So the filed rule reads "not through C" on
both seeds, and our filed prediction of "reuses C" missed. The twin reads "not through C", as filed.

## What we were trying to test

The definition we read under is Jain, Kirk, Lubana and colleagues' 2024 one, as adopted in `decisions.md` rows 81 and
87, with nothing added. It has three parts.

- **Possession.** A linear readout trained on ordinary Pile text, never on task windows, recovers the capability on the
  task's inputs, and does clearly better than a control readout trained on random labels.
- **Small function and carrying.** The fine-tuned model's answer is a simple function of that readout, and it is right
  on words it never trained on.
- **Causal check.** Removing the readout's direction breaks the fine-tuned behaviour, and spares a twin fine-tune that
  uses the same marker but cannot use the capability.

The capability is "the previous word is capitalised". We took the 400 most frequent words that exist as single tokens
in both cases, such as " The" and " the", and split them by pair into 300 training pairs and 100 held-out pairs. So no
held-out word was seen in either case during training. Each task window is a 41-token Pile window with a pool word
just before a fixed marker token, the trigger, and the label word after it. The marker is the token " \+", and the two
label words are " carpet" and " camel", drawn once from a fixed list of nouns.

- **Real task.** Say label A if the word before the trigger is capitalised, B if not.
- **Twin.** Same trigger, same two labels, but each word form gets A or B from a seeded shuffle. No feature of the
  original model predicts it, so the twin can only memorise. It rules out that any fine-tune at this trigger looks like
  reuse to our tests.

Each arm was fine-tuned with the recipe of an earlier copy task, two seeds, until accuracy on training words reached
0.9. The real arm got there at step 30, the twin at step 300.

The possession readout is a ridge regression on block $l$'s output residual, $\hat c_l(x) = w_l^\top x$, trained on
Pile positions whose previous token is a training-pair word, read as capitalised when the score exceeds one half. Its
control is the same regression with labels given by one random bit per word pair, following Hewitt and Liang. A
readout that only memorises words scores 0.5 on held-out pairs; a readout of case scores near 1.

The causal check removes the readout's unit direction $u = w_{l^*}/\|w_{l^*}\|$ from block $l^*$'s output residual at
the trigger position only, $x \mapsto x - u u^\top x$, where $l^*$ is the first block whose readout passes. It reports
the share of the label gain removed: with $m$ the label log-odds toward the window's own label and $G$ the fine-tune's
mean gain in $m$ over the original model,

$$\text{share} = \frac{\operatorname{mean}(m_{\text{state}} - m_{\text{state, direction removed}})}{G}.$$

A share of 1 means the whole gain goes with the direction, 0 means none of it does. The same share with a random unit
direction in the same place is the do-nothing reference.

## What we expected, and why

We expected possession to pass before the last block, since case is a feature every language model must track. We
expected the real fine-tune to carry to unseen words as a threshold on the case readout. A rule that holds across
words is learned fastest by reading a feature already there. We
expected removing the case direction to take at least half the real gain away, and almost none of the twin's. The twin
should fail carrying and small function, both near 0.5, because its labels are random across words.

## What we observed

### Does the original model hold the case of the previous word?

![possession](../../experiments/case_label/figures/possession.png)

**Reading it.** Yes, at block 1, the second of four. The case readout labels held-out task words at the trigger with
accuracy 0.994, while the random-label control reads 0.524, so the gap is 0.469 against the bar of 0.30. Blocks 0 and 2 fall short of the 0.90 accuracy bar, block 2 only just, and
the last block does not count. So the case of the
previous word is most cleanly readable at the trigger after the second block, and $l^* = 1$.

| block | case readout accuracy | control accuracy | gap | passes |
|---|---|---|---|---|
| 0 | 0.683 | 0.476 | 0.207 | no |
| 1 | 0.994 | 0.524 | 0.469 | yes |
| 2 | 0.885 | 0.464 | 0.421 | no, accuracy |
| 3 | 0.763 | 0.479 | 0.283 | last block, not counted |

The original model, unchanged, labels held-out windows right 0.521 of the time on the real task and 0.497 on the twin.
Removing the case direction at block 1's trigger position from the original model changed its next-token loss there by
0.003 nats, against 0.001 for a random direction, from a loss of 3.43.

### Does the fine-tune carry to unseen words, as a threshold on the case readout?

![carrying](../../experiments/case_label/figures/carrying.png)

**Reading it.** Yes for the real task on both seeds, no for the twin, as filed. The real fine-tunes label held-out
words right 0.998 and 0.996 of the time, against 0.52 for the original model. A single threshold on the original
model's case readout, fitted on training words, agrees with their labels on 0.993 and 0.992 of held-out windows. The
twins sit at 0.51 to 0.52 on both, at chance, though they reached 0.98 on their training words. One caveat: once the
readout reads case at 0.99 and the model labels by case at 0.99, the agreement must be near 0.99 whatever the model
computes inside. So this test cannot tell reading the feature apart from computing case some other way.

| state | scored step | train-word accuracy | held-out accuracy | agreement with threshold | summed update norm |
|---|---|---|---|---|---|
| real, seed 0 | 30 | 0.998 | 0.998 | 0.993 | 4.60 |
| real, seed 1 | 30 | 0.999 | 0.996 | 0.992 | 4.68 |
| twin, seed 0 | 300 | 0.978 | 0.506 | 0.511 | 13.33 |
| twin, seed 1 | 300 | 0.979 | 0.516 | 0.523 | 12.61 |

### Does removing the case direction take away the fine-tune's label gain?

![removal](../../experiments/case_label/figures/removal.png)

**Reading it.** No. Removing the case direction at block 1's output at the trigger took 0.083 and 0.091 of the real
fine-tunes' label gain, far under the filed bar of 0.50. The gain itself is large, 12.8 and 12.0 nats, so about one
nat of it went with the direction. For the twin the share was 0.001 and 0.003, and a random direction in the same place
took at most 0.0006 on every state. So the direction matters to the real fine-tune about thirty times more than a
random one does, but it carries a small part of the behaviour. The filed verdict for the real task is "not through C".

| state | label gain, nats | share removed, case direction | share removed, random direction | verdict as filed |
|---|---|---|---|---|
| real, seed 0 | 12.83 | 0.083 | 0.0004 | not through C |
| real, seed 1 | 12.04 | 0.091 | 0.0006 | not through C |
| twin, seed 0 | 10.23 | 0.001 | 0.0003 | not through C |
| twin, seed 1 | 8.80 | 0.003 | 0.0005 | not through C |

We do not know why the share is low. Candidate explanations, none tested:

- The case may still be present at the word's own position, one before the trigger, which the removal leaves alone.
  Later attention heads at blocks 2 and 3 could read it again from there.
- Case is carried in more than one direction at block 1, and the ridge readout's single direction is not the one the
  fine-tune reads, or not the only one.
- The fine-tune reads case at a later block, rebuilt there after the removal, or builds its own case detector from the
  word's identity.
- The removal is one direction, at one block, at one position: a rank-1 edit, the weakest removal the definition allows.

### Beside the verdict: pruning and reverse fine-tuning

![beside](../../experiments/case_label/figures/beside.png)

**Reading it.** No rule reads these; they are reported as filed. Zeroing the 64 most important MLP units of the real
fine-tune lowered held-out label accuracy only to 0.91 and 0.95, while 64 random units left it at 0.98 or above. So no
small set of MLP units carries the behaviour. Fine-tuning back toward the original next tokens brought the real
fine-tunes' held-out accuracy down to 0.6 within 50 and 60 steps, slower than the 30 steps it took to learn. The twins
start at chance on held-out words, so their reverse fine-tune reads 0 steps and says nothing.

| state | top-1 | top-4 | top-16 | top-64 | lowest of 5 random, K = 64 | reverse steps to 0.6 |
|---|---|---|---|---|---|---|
| real, seed 0 | 0.998 | 0.934 | 0.924 | 0.914 | 0.975 | 50 |
| real, seed 1 | 0.996 | 0.996 | 0.961 | 0.953 | 0.986 | 60 |
| twin, seed 0 | 0.506 | 0.506 | 0.505 | 0.505 | 0.498 | 0, starts below |
| twin, seed 1 | 0.514 | 0.514 | 0.511 | 0.504 | 0.513 | 0, starts below |

## What we learned

The definition as adopted does not read "reuses" on the pair built to its own shape, even though the real fine-tune
looks like reuse on every test but the causal one. Possession and carrying both read the way reuse would, and the twin
separates from the real task on both, so the apparatus does tell a feature-based fine-tune from a memorising one. The
causal check, as filed, is the test that decides, and with a rank-1 removal at one position it removes about a tenth of
the gain. For `GOALS.md`'s question of new capabilities against strengthened ones, this pair therefore gives no
outside "reuse" label yet. Whether the stronger removal the user approved changes that is the next run. Subject number,
the second pair, waits on it.

## How compelling is the evidence

- Two seeds per arm; the two seeds agree to within 0.01 on every rule number.
- One task pair, one model, one marker token, one block for the removal.
- The small-function test is blind when the readout and the labels are both near perfect: it agrees at 0.99 by
  arithmetic. It separates real from twin, not reading from recomputing.
- The removal is blind to case carried at other positions, other blocks or other directions. Its noise floor, the
  random direction, is at most 0.0006, so the 0.08 to 0.09 share is far above noise and far below the bar.
- The held-out set has 1,600 windows over 100 word pairs; the fit set 1,024 windows of training words.

## Surprises

The filed prediction that the real fine-tune "reuses C" missed on both seeds: the removal share was 0.08 and 0.09
against a bar of 0.50, while every other number read as predicted. We predicted that removing the case direction
would take at least half the label gain, and it took under a tenth. We cannot explain it yet; the candidates above are
untested.

## What next

- Amendment 1, a stronger removal the user approved: take the direction out of every position at every block from
  block 1 on, each block with its own readout. It is being coded now. If the share then reaches one half, the low
  reading here was the removal's reach; if not, the fine-tune does not run through this readout.
- Subject number, the second pair, runs only once case label reads cleanly, per the user's order.

## Scorecard

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| P0 | the original model holds case before the last block | accuracy at least 0.90 and gap at least 0.30 | block 1: 0.994, gap 0.469 | pass, l* = 1 |
| do-nothing | a random direction removes almost no gain | share at most 0.10 on every state | at most 0.0006 | pass |
| P1 real | real fine-tune right on held-out words | at least 0.90 | 0.998, 0.996 | pass |
| P2 real | labels are a threshold on the readout | agreement at least 0.90 | 0.993, 0.992 | pass |
| P3 real | removing the direction removes the gain | share at least 0.50 | 0.083, 0.091 | **miss** |
| real verdict | "reuses C" on both seeds | all of P1-P3 | P3 misses | **"not through C", filed outcome missed** |
| P1 twin | twin at chance on held-out words | at most 0.65 | 0.506, 0.516 | as filed |
| P2 twin | twin not a threshold on the readout | at most 0.65 | 0.511, 0.523 | as filed |
| P3 twin | removal spares the twin | at most 0.20 | 0.001, 0.003 | as filed |
| twin verdict | "not through C" on both seeds | any of P1-P3 misses | P1 and P2 miss | **"not through C", as filed** |

## Departures from the literature

- The readout is a binary ridge regression on one block's residual at the trigger, not Jain et al.'s probe for their
  synthetic grammar's capability, since our task has no such grammar. Same role: a linear readout trained off-task.
- Carrying to held-out words is our addition from `decisions.md` row 87, a dated departure the user made; Jain et al.
  do not require it.
- The direction-removal check is our addition from row 81; Jain et al. use pruning and reverse fine-tuning, which run
  here beside it without a rule.
- The fine-tune is our earlier copy task's recipe, Adam at 3e-5 with a KL term to the original model, not Jain et
  al.'s recipe for their models.
- The control readout follows Hewitt and Liang 2019 in giving each word pair a random label, scored on held-out pairs.

## Not run

- Amendment 1, the stronger removal: the direction out of every position at every block from block 1 on, each
  block's own readout. The user approved it on 2026-11-06, answering ask 3. It is coming, and its result will be added
  to this report.
- Subject number, the second pair; it waits on the decision above.
- Removal at blocks other than block 1, or at the pool word's own position.
- More than two seeds; a second marker token; other label words.
- The route-split readings, which subcomponents Γ switches on, at this trigger.
- An independent check of the analysis.

## Provenance

- Results: `experiments/case_label/results/stage0.json`, `stage1.json`, `stage2.json`, logs `stage{0,1,2}.out`.
- Update states: `experiments/case_label/results/large/dW_{real,twin}_s{0,1}.pt`, bf16; sha256 in `stage1.json`.
- Code: `experiments/case_label/run.py`, removal in `removal_edit`; rules in `src/commute/case_label.py`; figures by
  `experiments/case_label/plot.py`.
- Jobs 54048, 54052, 54054, MATS cluster debug QoS, one L40, 10:53 to 11:03 UTC, about 10 minutes, $0. Peak host
  memory 6.1 GB.
