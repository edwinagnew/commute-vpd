# Does a fine-tune that learns to count a new letter run through the count the model already had?

E55 Stage A, reuse line, 2026-10-29. Plan `reports/plans/2026-10-29_plan_reuse-stage1.md`, Revision 1. Predictions
`experiments/pcfg_anchors/results/predictions.json`, filed four times, the last before any fine-tune. Notebook
`notebooks/E55_pcfg_anchors.ipynb`, figures `experiments/pcfg_anchors/figures/` from `plot.py`.

We fine-tuned a small model that counts only the letter a to count the letter b, and asked whether each new answer runs through the old count, judged without any decomposition.
Fast fine-tunes never used the old count, while slow fine-tunes trained mostly or only on rows the old count answers used it after a thousand steps, and by the end neither check read any of them as using it.
The reuse line now has a working ground truth outside VPD, but it marks only two early states as reusing the old count, so scoring VPD against it would be lopsided.

## The answer

The two checks work as a ground truth on this grammar. Swapping the count in moves the answer on every model that
uses it and never on the nulls. Fast fine-tunes never ran through the count of a. The slow fine-tunes with 80% and
with all spurious rows ran through it after a thousand steps. By the end the first had learned b and left the old
count, and the all-spurious ones read neither, so one filed prediction missed. None was reversed. The count of b
decodes from the base model above shuffled labels, narrowly.

## What we were trying to test

`GOALS.md` asks, if no simple map works, whether post-training teaches new capabilities or strengthens existing
ones. Until now our labels "existing" and "new" came from VPD itself. This stage builds labels that need no
decomposition, on a task where we choose whether the model already has the capability.

**The task.** A sequence is a prefix, 250 tokens of grammar text over the letters a, b and c, and a query. The
Counter prefix `C a 40` asks how many a's are in the last 40 text tokens; `C b 40` asks the same for b. The model
predicts the answer at the `=` token. Every read and every swap below happens at that position.

**The models.** A minGPT of 6 blocks, width 192, about 3M parameters, pretrained for 50K steps on Jain's five task
families. Three pretraining mixes fix what each model can do before any fine-tune:

| model | Counter operand mix | what it can do |
|---|---|---|
| LOW | a almost always, b one time in a thousand, never c | counts a; asked for b, it answers its count of a |
| HIGH | a half, b three tenths, c a fifth | counts a and b |
| NEVER | b only | counts b; never saw the `C a 40` prefix |

**The fine-tunes.** Each fine-tune trains a model to answer `C b 40` for 10K steps, with states saved at 1K and 10K.
Spurious rows are training rows whose `count(b)` keeps its natural value and whose count of a in the window is set
to `count(b) - 1` by swapping a's and c's, so on them answering `count(a) + 1` is right. The text holds only a, b
and c, so the three window counts sum to 40 on every row, natural or spurious.
The arms follow Table 6 of the paper:

| arm | base | learning rate | spurious share | what it rules out |
|---|---|---|---|---|
| N1, N2, N3 | LOW | 1e-4 | 0, half, 80% | fast learners: count b for real, the "new" side |
| M, M2, M3 | LOW | 1e-5 | 0, half, 80% | slow learners: may pass through count(a) + 1 on the way |
| W1, W2 | LOW | 1e-5, 1e-4 | all | count(a) + 1 is right on every training row: the "existing" side |
| R | HIGH | 1e-5 | 0 | a base that already counts b: the answer should run through its count of b |
| N1, W1 seed 1 | LOW | as above | as above | fine-tune noise |

**The two checks.** For a fine-tuned state and a count, on natural inputs where `count(b)` differs from `count(a) + 1`:

- **Behaviour.** For the count of a, the share of answers equal to `count(a) + 1`; for the count of b, the share equal to `count(b)`.
  0 means the answer never takes that value; 1 means always.
- **Patch.** We build 1,000 pairs of inputs with the same value of the other count and different values of this one.
  We copy the donor's residual stream at `=` after block 5 into the run on the first input. The patch is the share of
  pairs whose answer becomes the donor's own answer, where the two answers differ. 0 means the swapped count
  is never used at that point; 1 means the answer always follows it.

**The bars.** Each check has a null and a positive reference measured on the pretrained models before any
fine-tune. The upper bar is halfway from the null to the positive. The lower bar is the null plus the larger of three
standard deviations and a tenth of the null-to-positive gap. A state **goes through** a count when both checks clear
the upper bar, **does not go through** it when both sit under the lower bar, and reads **neither** otherwise.

| check | null | positive | lower bar | upper bar |
|---|---|---|---|---|
| behaviour, count of a | NEVER on `C b 40`: 0.000 | LOW on its own count-a task: 0.9998 | 0.100 | 0.500 |
| patch, count of a | NEVER on `C b 40`: 0.000 | LOW on `C a 40`: 0.966 | 0.097 | 0.483 |
| behaviour, count of b | LOW on `C b 40`: 0.269 | HIGH's count-b accuracy: 1.000 | 0.342 | 0.635 |
| patch, count of b | LOW on `C a 40`: 0.000 | HIGH on `C b 40`: 1.000 | 0.100 | 0.500 |

The behaviour null for the count of b is not zero: LOW answers `count(a)` when asked for b, and on the qualifying inputs
that is right whenever the two counts agree.

**Possession.** A linear probe reads the count from the residual at `=` after each block. Its margin is its held-out
accuracy minus the best of its controls, a probe trained on shuffled labels and, for the count of a, NEVER's probe.
A model possesses a count when, at some block, the margin is at least three standard deviations over probe seeds and
at least half of LOW's best count-a margin.

## What we expected, and why

- The pretrained models would read as the plan tabled: LOW counts a and not b, HIGH counts both, NEVER counts b; the
  patch moves the answer on the positive controls and not on the nulls. The behaviour rows follow Table 6; the patch
  rows were new on this grammar, so the lean was only medium.
- The fine-tunes would reproduce Table 6's count-b accuracies within 15 points in at least seven of nine cells. The
  filing noted before the fine-tunes that the two all-spurious cells could miss by construction. A model answering
  `count(a) + 1` scores about 18% on natural rows, while the paper prints 0 and 1.9.
- The all-spurious fine-tunes would go through the count of a at both states, N1 would not go through it at either,
  and R would go through its count of b. The lean was medium, because the paper says only what the all-spurious
  fine-tunes do not answer.
- LOW would possess the count of b, at even odds: it almost never answers for b, but every task family counts tokens.

## What we observed

### Do the checks work on the pretrained models?

![The patch by block on the pretrained models](../../experiments/pcfg_anchors/figures/controls.png)

**Reading it.** Yes. The figure plots, for each block whose output we swap, the share of pairs whose answer becomes the
donor's. On the two models asked for the count they use, the share rises from near zero to 0.97 and 1.00 at block 5.
On the two nulls it is zero at every block: there the swapped count is one the model does not answer with. The
earliest block within three standard deviations of the best, excluding the last block, is block 5 for both counts.
Every fine-tuned state is read there. One caveat is real. Block 5 is next to the output, so the patch there is
close to a second behaviour reading rather than a test of where the count is first computed.

Behaviour on the pretrained models, 4,000 rows each, beside the score of guessing:

| model | count-a accuracy | count-b accuracy | majority-answer score | score of answering count(a) when asked for b |
|---|---|---|---|---|
| LOW | 0.99975 | 0.2215 | 0.348 | 0.2215 |
| HIGH | 0.99975 | 1.000 | 0.348 | 0.2215 |
| NEVER | 0.2175 | 1.000 | 0.348 | 0.2215 |

LOW's count-b accuracy equals, to four digits, the score of answering `count(a)`: asked for b, it answers its count of
a. A second, independent number agrees. An exact count-of-a answer is never right on the 0.1765 of rows where
`count(b)` equals `count(a) + 1`, so on the qualifying inputs it scores 0.2215 / (1 − 0.1765) = 0.269. That is
LOW's measured behaviour null for b. Table 6 prints 5.2% for this cell, probably because our window counts, which sit
between 9 and 17, agree more often than the paper's. We did not measure the paper's spread.

Patch at every block, mean of three sets of 1,000 pairs:

| reference | block 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| LOW, count a, `C a 40` | 0.000 | 0.012 | 0.095 | 0.467 | 0.966 | 1.000 |
| HIGH, count b, `C b 40` | 0.045 | 0.152 | 0.151 | 0.646 | 1.000 | 1.000 |
| NEVER, count a, `C b 40` | 0 | 0 | 0 | 0 | 0 | 0 |
| LOW, count b, `C a 40` | 0 | 0 | 0 | 0 | 0 | 0 |

Behaviour nulls over three sets: NEVER's share of `count(a) + 1` answers 0, 0, 0; LOW's share of `count(b)` answers
0.268, 0.284, 0.255.

### Does the rebuilt grammar reproduce the paper's fine-tunes?

![Count-b accuracy at 10K against Table 6](../../experiments/pcfg_anchors/figures/table6.png)

**Reading it.** Yes, in seven of nine cells, the most the filing allowed. Every fine-tune with any natural rows
reaches 99.85% to 100% on count b, within six points of the paper. The two all-spurious fine-tunes miss. The slow
one scores 17.7%, equal to the score of always answering `count(a) + 1`, although by 10K it gives `count(a) + 1` on
only 0.40 of qualifying inputs. It gets its points on the rows where `count(b)` equals `count(a) + 1` and almost none
elsewhere. The fast one scores 26.0%, above that floor, where the paper prints 1.9%. Our fine-tunes also learn faster early. At 1K the slow learners read 26% to 48%;
the paper prints 1.4% to 33%. The early column was not filed.

| arm | ours at 1K, % | Table 6 at 1K | ours at 10K, % | Table 6 at 10K | within 15 at 10K |
|---|---|---|---|---|---|
| N1 | 99.8 | 98.9 | 100.0 | 100 | yes |
| N2 | 99.35 | 90.2 | 100.0 | 99.9 | yes |
| N3 | 95.8 | 64.9 | 100.0 | 100 | yes |
| M | 48.3 | 32.6 | 99.95 | 99.8 | yes |
| M2 | 35.7 | 12.7 | 99.95 | 99.1 | yes |
| M3 | 25.7 | 1.4 | 99.85 | 93.9 | yes |
| W1 | 17.75 | 0 | 17.675 | 0 | no, off by 17.7 |
| W2 | 18.1 | 0 | 25.95 | 1.9 | no, off by 24.05 |
| R | 100.0 | 99.9 | 100.0 | 99.9 | yes |

### Which fine-tunes run through the old count of a?

![Every fine-tuned state on the two checks](../../experiments/pcfg_anchors/figures/labels.png)

**Reading it.** Fast fine-tunes never go through the count of a. Slow ones with many spurious rows go through it
early, and by 10K neither check reads them as going through it. Every fast fine-tune with natural rows sits under
both lower bars at both states. At 10K every fine-tune that learned to count b sits at exactly zero on both checks, and goes through the count of b.

The slow fine-tune with 80% spurious rows goes through the count of a at 1K. It gives `count(a) + 1` on 0.63 of
inputs and follows a swapped count of a on 0.59 of pairs. By 10K it counts b at 99.85% and reads not through. The
slow half-spurious fine-tune at 1K follows the swap on 0.52 of pairs, over the upper bar of 0.483. It answers
`count(a) + 1` on only 0.43, so it reads neither. The plan expected both of these 1K states to go through.

The all-spurious fine-tunes start where the filing expected. After 1K steps the slow one gives `count(a) + 1` on 0.88
of inputs. It follows a swapped count of a on 0.72 of pairs, and its seed copy reads 0.90 and 0.76. By 10K both
copies fall to about 0.40 and 0.36, between the bars, so they read neither. The fast all-spurious fine-tune reads
neither at both states. At 1K its behaviour is 0.488, just under the upper bar of 0.500. HIGH's fine-tune goes through
its count of b, as expected. So every slow fine-tune with 80% or more spurious rows goes through the old count at
1K and no longer reads as going through it at 10K, not only the all-spurious one.

What the all-spurious fine-tunes do at 10K is not settled by the checks. A swap at the last block shows what the
answer depends on at all. For the slow all-spurious fine-tune at 1K, a swapped count of a moves the answer on 0.94 of
pairs and a swapped count of b on 0.17. At 10K the two shares are 0.64 and 0.57: the answer depends on both counts.

One reading would have the answer built from the count of c, which fits every spurious row. For the slow
all-spurious fine-tune that reading is ruled out by the sum of 40. We sampled 65K qualifying natural windows, with no
model. Any answer that depends on the count of c alone and equals `count(b)` on at most 0.01 of them gives
`count(a) + 1` on at most 0.105. The slow all-spurious state reads 0.40 and at most 0.001, at both seeds.

One rule fits both of its numbers: answer the midpoint of `count(a) + 1` and `count(b)`, rounded towards
`count(a) + 1`. On the same windows it reads 0.41 and 0.00, and it is exact on every spurious row. That rule reads the
count of a on every input, through a function the behaviour check is blind to. It fits, but it is untested. The fast
all-spurious state at 10K, at 0.29 and 0.10, is not excluded by the bound.

| state | behaviour a | patch a | reads, count of a | behaviour b | patch b | reads, count of b |
|---|---|---|---|---|---|---|
| N1 1K | 0.000 | 0.000 | not through | 0.993 | 0.883 | through |
| N1 10K | 0.000 | 0.000 | not through | 1.000 | 0.573 | through |
| N1 seed 1, 1K | 0.000 | 0.000 | not through | 0.993 | 0.924 | through |
| N1 seed 1, 10K | 0.000 | 0.000 | not through | 1.000 | 0.650 | through |
| N2 1K | 0.002 | 0.005 | not through | 0.980 | 0.971 | through |
| N2 10K | 0.000 | 0.000 | not through | 1.000 | 0.827 | through |
| N3 1K | 0.008 | 0.066 | not through | 0.926 | 0.859 | through |
| N3 10K | 0.000 | 0.000 | not through | 1.000 | 0.858 | through |
| M 1K | 0.118 | 0.437 | neither | 0.467 | 0.395 | neither |
| M 10K | 0.000 | 0.000 | not through | 0.999 | 0.818 | through |
| M2 1K | 0.430 | 0.519 | neither | 0.221 | 0.389 | neither |
| M2 10K | 0.000 | 0.000 | not through | 0.999 | 0.988 | through |
| M3 1K | 0.631 | 0.594 | through | 0.099 | 0.328 | neither |
| M3 10K | 0.000 | 0.000 | not through | 0.994 | 0.968 | through |
| W1 1K | 0.879 | 0.715 | through | 0.001 | 0.104 | neither |
| W1 10K | 0.399 | 0.361 | neither | 0.001 | 0.429 | neither |
| W1 seed 1, 1K | 0.903 | 0.764 | through | 0.001 | 0.073 | not through |
| W1 seed 1, 10K | 0.399 | 0.363 | neither | 0.000 | 0.427 | neither |
| W2 1K | 0.488 | 0.414 | neither | 0.004 | 0.358 | neither |
| W2 10K | 0.292 | 0.344 | neither | 0.102 | 0.377 | neither |
| R 1K | 0.000 | 0.000 | not through | 1.000 | 1.000 | through |
| R 10K | 0.000 | 0.000 | not through | 1.000 | 1.000 | through |

Every state built 1,000 pairs for each patch, so no patch was void. The filed null did not occur: on no state
was one check over its upper bar and the other under its lower bar. A softer mismatch is visible in the right panel. At 10K the all-spurious states almost never answer `count(b)`.
Yet a swapped count of b moves their answer at block 5 on 0.38 to 0.43 of pairs, as the midpoint rule would.

Two further readings were not filed. N1's patch for the count of b falls from 0.88 at 1K to 0.57 at 10K.
Its seed copy falls from 0.92 to 0.65, while both still answer `count(b)` every time. That is consistent with the
count of b reaching the answer later than block 5 on more inputs, which is untested. Either way N1's label sits
closer to its upper bar. Second, every N and M fine-tune also changed LOW's
answer to its own count-a task. Accuracy there fell from 0.99975 to 0.2175 at 10K, the same as NEVER's, which counts b
whatever it is asked. HIGH's fine-tune kept its count-a accuracy at 0.974. One counter re-bound to the new letter would
look like this, and the rule reads that as not going through the count of a, as the plan stated.

Stage C would score the sixteen states from LOW at the first seed. Two go through the count of a, both at 1K: the
slow 80%-spurious and the slow all-spurious arms. Nine do not and five read neither, where the plan expected six,
eight and two: the plan leaned on both all-spurious arms at both states and on two slow arms at 1K going through.

### Does LOW already hold a count of b?

![Probe margin by block](../../experiments/pcfg_anchors/figures/probe.png)

**Reading it.** Yes by the filed rule, narrowly, and the reading behind the rule is weaker than the pass. The
figure plots the probe margin by block. LOW's count-b margin is 0.277 at block 2, over the possession bar of 0.260 by
about one and a half standard deviations. It clears the bar at block 3 too, then falls to 0.18 by block 6. Its
count-a margin, the positive control, rises to 0.52. HIGH, which was trained on count b, reaches 0.65.

The two margins of LOW subtract different controls. The count-a margin subtracts NEVER's count-a probe as well as
shuffled labels. NEVER never had a count-a task, yet it decodes count a at 0.44 to 0.52 in blocks 2 to 6, against
about 0.34 for shuffled labels. The count-b margin subtracts shuffled labels only, because no model in this design
lacks a count-b task in the way NEVER lacks a count-a one. LOW's count-b probe peaks at 0.626, about 0.11 above what
NEVER reaches for a count it was never asked for. Against that kind of control the gap is under half the bar. Also,
the count of b can be guessed from the count of a alone at 0.40 on natural windows, so part of the margin over
shuffled labels can come from LOW's count of a. So the count of b decodes from LOW's residual above shuffled labels,
by less than a model with no count-a task decodes the count of a. A probe also cannot say whether the count is used.

| model, count | best block | probe accuracy | majority-class rate | margin | standard deviation over 3 probe seeds |
|---|---|---|---|---|---|
| LOW, a | 6 | 1.000 | 0.341 | 0.520 | 0.0088 |
| HIGH, a | 6 | 1.000 | 0.341 | 0.520 | 0.0088 |
| HIGH, b | 4 | 0.994 | 0.348 | 0.646 | 0.0050 |
| LOW, b | 2 | 0.626 | 0.348 | 0.277 | 0.0108 |

Possession bar: half of LOW's best count-a margin, 0.260; voiding line 6 standard deviations of that margin, 0.053.

## What we learned

The reuse line has a ground truth that owes nothing to any decomposition, and its two checks never point opposite ways on
all 22 states read. On this grammar a fast fine-tune that learns a new count never runs its answer through the old
count. Every slow fine-tune with 80% or more spurious rows runs through it after a thousand steps. By the end it has
either learned the new count or gives an answer neither check reads as the old count. Training only on rows where the
old count gives the answer was meant to be the clearest "existing" anchor. Whether those fine-tunes still use the count
of a at 10K, through a function other than `count(a) + 1`, the checks cannot say. So "existing" labels are scarce: two
of the eleven labelled states, both at 1K. That leaves any method scored against them, VPD included, with a lopsided
truth set. If the all-spurious states at 10K turn out to use the count of a through a midpoint, the scarcity weakens.

## How compelling is the evidence

- **The checks.** The controls are clean: nulls at zero in every set and block, positives at 0.97 and 1. Each
  fine-tuned state is read once, on one set of 1,000 pairs and 1,000 inputs shared by every state. The binomial
  standard deviation of one reading is about 0.016 near 0.5 and about 0.009 near 0.1. Two cells sit within it of a
  bar. The fast all-spurious state at 1K is 0.012 under the upper behaviour bar for the count of a, about 0.75 sd.
  Its patch is 0.069 under its bar, so its label does not depend on that cell. The slow all-spurious state at 1K reads
  0.104 and 0.073 on the count-b patch over its two seeds, either side of the lower bar. The next closest, the slow
  no-spurious state at 1K, sits about 2 sd over the lower behaviour bar, and its patch alone makes it "neither". No
  count-a label, the labels Stage C would use, depends on a cell within noise.
- **What the behaviour check is blind to.** It counts answers equal to one value; an answer that uses the count
  through another function reads low, as the midpoint rule would. Its positive reference for the count of a is LOW's
  plain count-a accuracy on `C a 40`. That is a different quantity from the fine-tuned reading, which is the share of
  answers equal to `count(a) + 1` on qualifying `C b 40` inputs. No positive control of that reading itself was run;
  the slow all-spurious state at 1K, at 0.88, shows only that it can read high.
- **What the patch is blind to.** It reads one position and one block. Block 5 is next to the output, so it tests
  that the count is present late, not where it is computed. A count computed after block 5 from text positions
  reads low. The three counts sum to 40 on every input. So at a fixed count of b, changing the count of a changes the
  count of c, and at a fixed count of a, changing b changes c. Neither patch can tell a count of c from the count it
  swaps, and the count-b labels of R and N1 carry the same ambiguity. Only a mechanism-level test separates them. The
  count-b patch null is LOW on `C a 40`, not on the fine-tunes' own prefix `C b 40`.
- **What the probe is blind to.** Use: a decodable count need not be one the model computes with. Taking the best of
  six blocks inflates false positives slightly. LOW's count-b margin clears its bar by about one and a half
  standard deviations. Its controls are uneven: the count-a margin subtracts a model with no count-a task, while
  the count-b margin subtracts only shuffled labels. Part of the count-b margin can come from the count of a, which
  predicts the count of b at 0.40 on its own.
- **Seeds.** One pretraining seed per mix. Two fine-tune seeds for N1 and the slow all-spurious arm, which agree on
  every label for the count of a. One task family and one window, 40 tokens.

## Surprises

- The all-spurious fine-tunes stop reading as going through the count of a between 1K and 10K steps, while their
  accuracy on spurious test rows stays at 1: they still answer every spurious row right at 10K. A filed prediction
  missed for this reason; it was not reversed, since no all-spurious state reads "does not go through".
- Every fine-tune that learned to count b from LOW also wiped out LOW's answer on its own count-a task.
- LOW's count-b accuracy before any fine-tune sits exactly on the score of answering its count of a.

## What next

1. **Decide whether instrument scoring goes ahead on these labels.** The plan's decision tree covers a pass, a
   reversal and the null, but not a miss without reversal. Eleven states are labelled, two of them "existing", so a
   method must agree on ten to pass. If that is judged too lopsided, more "existing" anchors come first.
2. **Read the per-row answers of the all-spurious states at 10K against the counts of a and b**, $0 on the saved
   states; E55 Amendment 1 is being written for this. If the answers follow the midpoint of `count(a) + 1` and
   `count(b)`, those states still use the count of a through another function, and the truth set may gain
   "existing" anchors. If they follow no function of the two counts, the "neither" labels stand.
3. **Read the all-spurious arms between 1K and 10K** to find when they stop reading as going through the count of
   a, so that more "existing" states can be saved.

## Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| PA1 behaviour | LOW counts a, not b; HIGH counts both; NEVER counts b | LOW a ≥ 0.95; LOW b ≤ swap floor + 0.10; HIGH a, b ≥ 0.95; NEVER b ≥ 0.95 | 0.99975; 0.2215 ≤ 0.3215; 0.99975, 1.000; 1.000 | pass |
| PA1 count-a probe | LOW's count-a margin is clear; HIGH's is at least half of it | m₊ ≥ 6 s₊; HIGH ≥ 0.5 m₊ | 0.520 ≥ 0.053; 0.520 ≥ 0.260 | pass |
| PA1 count-b probe | HIGH's count-b margin is at least half of LOW's count-a margin | ≥ 0.5 m₊ | 0.646 ≥ 0.260 | pass |
| PA1 patch positive | the patch moves the answer on the positive controls | ≥ 0.8 at block 5 | 0.966, 1.000 | pass |
| PA1 nulls | the patch and behaviour nulls stay near zero | each ≤ 0.05 | patch 0, 0; behaviour 0 | pass |
| PA2 | count-b accuracy at 10K within 15 points of Table 6 | ≥ 7 of 9 cells | 7 of 9; the two all-spurious cells miss by 17.7 and 24.05 | pass, at the cap the filing anticipated |
| PA3 | all-spurious arms through count a at 1K and 10K; N1 not through at both; R through count b | all 7 readings | 4 of 7: slow all-spurious 1K through; its 10K neither; fast all-spurious 1K and 10K neither; N1 both not; R through | **miss**, not reversed |
| PA4 | LOW possesses the count of b | m ≥ 3 s and m ≥ 0.5 m₊ at some block | block 2: 0.277 ≥ 0.032 and ≥ 0.260 | pass by its rule, narrowly; the count-b margin lacks the no-task control the count-a margin has |
| PA3 null | the two checks point opposite ways on ≥ 3 of 17 states | count | 0 of 17 | did not occur |
| Stage C pass bar | agreements needed with 11 labelled states | smallest a with binomial tail ≤ 0.05, raised to majority + 1 | a = 9, raised to 10 by the majority of 9 "not through" | for Stage C |

All A1 stops passed; no bar was void. PA3's thresholds are the A2 bars filed before any fine-tune; labels were read
by `interchange.label` with strict inequalities.

## Not run

- Stage B, lowercase on the 67M; the free part of Stage C, pruning, reverse fine-tuning and steering; the gated part, a VPD decomposition
  of LOW, gated.
- Per-row answers of the all-spurious states at 10K against the counts of a and b, which would test the midpoint
  rule; E55 Amendment 1. They were not saved, so the mixed answers cannot be read from the results files. States
  between 1K and 10K.
- A mechanism-level test that separates the count of c from the counts of a and b, for either patch.
- A positive control of the fine-tuned behaviour reading itself, the share equal to `count(a) + 1` on qualifying
  `C b 40` inputs. A count-b patch null on `C b 40`, the fine-tunes' own prefix.
- Resampling of each fine-tuned state's behaviour and patch readings; each was read once.
- A patch at the text positions, or at any position other than `=`.
- A no-task control for the count-b probe, the analogue of NEVER for the count of a; probes on the fine-tuned states.
- A second pretraining seed of any mix; fine-tune seeds beyond N1 and the slow all-spurious arm; the paper's 200K
  pretraining; Table 6's spurious test column beyond the accuracy reported in the results files.

Provenance: `experiments/pcfg_anchors/results/{a0,a1_LOW,a1_HIGH,a1_NEVER,a1c_patch,a1c_probe}.json` and
`a2_<arm>.json` for N1 N2 N3 M M2 M3 W1 W2 R N1s1 W1s1; code `experiments/pcfg_anchors/run.py`,
`src/commute/interchange.py`, `src/commute/pcfg.py`. MATS cluster `debug` QoS, one GPU per job: rehearsal 44244,
pretrainings 44268-44270, controls 44306 and 44307, fine-tunes 44328-44330; $0. Commits 6299401, 55da119, 87f77ce,
7e62cb4.

## Review responses

Corrections from `reports/reviews/2026-10-29_review_e55-stage-a-analysis.md` were applied on 2026-10-29: the headline
now matches the table, the count-c reading is ruled out for the slow all-spurious state, PA4's reading is hedged,
and the noise, blind-to and "Not run" items were added. One figure differs from the review. NEVER's count-a probe
accuracy, averaged over the three probe seeds, runs 0.44 to 0.52 over blocks 2 to 6, not 0.47 to 0.52: block 3 reads
0.44 (`a1c_probe.json` probes.raw.count_a NEVER.acc). The conclusion drawn from it is unchanged. The count-c bound and
the midpoint rule's 0.41 and 0.00 were re-run from the reviewer's scripts: `conly.py` reproduced 0.408 and 0.000
on 65,019 qualifying windows, and `conly3.py` the bound of 0.105 at a count-b share of at most 0.01.
